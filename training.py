"""Training, on Ray.

Separate from op.py because the two run in different places: op.py runs in the
operator's own process for every message, while this runs distributed and rarely.

The model is a seasonal profile over local hour of week (168 slots). Each slot
holds an exponentially recency-weighted mean of the hourly means of the target
series that fell in it. A slot with no history takes the overall weighted mean.

Assumption: the site keeps local time in Europe/Berlin. Hour of week is taken in
that zone, so daily and weekly load shapes stay aligned across the clock change.

provide_historic_data() is Operator Lib's reader over the platform's timeseries
store. It returns Ray Datasets, so the training below never holds the whole
history in one process.
"""

import datetime
import typing

import numpy as np
import pandas as pd
import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger


# How much history one training pass reads. A full year, so that every hour of
# the week has seen the seasons that the forecast will meet.
TRAINING_WINDOW = datetime.timedelta(days=365)

# The weight of an observation halves every this many weeks. Recent weeks count
# more, which follows slow drift in the household without a fitted trend.
HALF_LIFE_WEEKS = 4.0

LOCAL_TZ = "Europe/Berlin"
SLOTS = 7 * 24


class SeasonalLoadModel(PythonModel):
    """The model MLflow registers and op.py later loads.

    predict() receives the target timestamp and returns the profile value for its
    local hour of week. Keep it serialisable: MLflow stores this object.
    """

    def __init__(self, profile: typing.List[float], fallback: float) -> None:
        self.profile = [float(v) for v in profile]
        self.fallback = float(fallback)

    @staticmethod
    def slot_of(timestamp: datetime.datetime) -> int:
        local = pd.Timestamp(timestamp)
        if local.tzinfo is None:
            local = local.tz_localize("UTC")
        local = local.tz_convert(LOCAL_TZ)
        return int(local.weekday() * 24 + local.hour)

    def predict(self, context, model_input=None, params=None):
        # The pyfunc signature carries a context when MLflow calls it and not when the
        # model is called directly, so the payload is taken from whichever argument
        # holds it.
        payload = model_input if model_input is not None else context
        value = self.profile[self.slot_of(payload["timestamp"])]
        return value if value is not None else self.fallback


@ray.remote
def _hourly_sums(dataset) -> typing.Dict[int, typing.Tuple[float, int]]:
    """Per UTC hour: the sum and count of the values that fell in it.

    Sums and counts rather than means, so that an hour split across two batches
    is still one exact mean once the partial results are merged in the driver.
    The key is the hour as nanoseconds since the epoch, which pickles cleanly.
    """
    sums: typing.Dict[int, typing.Tuple[float, int]] = {}
    for batch in dataset.iter_batches(batch_size=65536, batch_format="pandas"):
        if "time" not in batch or "value" not in batch:
            continue
        times = pd.to_datetime(batch["time"], utc=True)
        values = pd.to_numeric(batch["value"], errors="coerce")
        frame = pd.DataFrame({"hour": times.dt.floor("h"), "value": values}).dropna()
        if frame.empty:
            continue
        grouped = frame.groupby("hour")["value"].agg(["sum", "count"])
        for hour, row in grouped.iterrows():
            key = int(hour.value)
            total, count = sums.get(key, (0.0, 0))
            sums[key] = (total + float(row["sum"]), count + int(row["count"]))
    return sums


def _fit_profile(sums: typing.Dict[int, typing.Tuple[float, int]]):
    """Weighted hour-of-week profile from the merged hourly means.

    Returns (profile of SLOTS values, overall weighted mean, hours used, slots
    that had at least one hour of history).
    """
    keys = sorted(k for k, (_, count) in sums.items() if count > 0)
    hourly = np.array([sums[k][0] / sums[k][1] for k in keys], dtype=float)
    index = pd.DatetimeIndex(pd.to_datetime(np.array(keys, dtype=np.int64), utc=True))

    # Ages are measured back from the newest hour in the window, which is the
    # training end, so the weights do not depend on the wall clock.
    age_weeks = np.asarray((index.max() - index) / pd.Timedelta(weeks=1), dtype=float)
    weights = 0.5 ** (age_weeks / HALF_LIFE_WEEKS)

    local = index.tz_convert(LOCAL_TZ)
    slots = np.asarray(local.weekday) * 24 + np.asarray(local.hour)

    fallback = float(np.sum(weights * hourly) / np.sum(weights))
    profile = np.full(SLOTS, fallback, dtype=float)
    filled = 0
    for slot in range(SLOTS):
        mask = slots == slot
        if mask.any():
            profile[slot] = float(np.sum(weights[mask] * hourly[mask]) / np.sum(weights[mask]))
            filled += 1
    return profile, fallback, int(len(hourly)), filled


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    """Read the history, fit the hour-of-week profile, and hand back a model."""
    with logger.trace("read history"):
        datasets = provide_historic_data(TRAINING_WINDOW)
    if not datasets:
        # Explicitly nothing rather than a model fitted on no data: returning None
        # leaves the previously registered model in place.
        return None

    with logger.trace("hourly sums"):
        resolved = [ray.get(d) if isinstance(d, ray.ObjectRef) else d for d in datasets]
        partials = ray.get([_hourly_sums.remote(d) for d in resolved])

    merged: typing.Dict[int, typing.Tuple[float, int]] = {}
    for partial in partials:
        for key, (total, count) in partial.items():
            t, c = merged.get(key, (0.0, 0))
            merged[key] = (t + total, c + count)

    if not any(count > 0 for _, count in merged.values()):
        return None

    with logger.trace("fit"):
        profile, fallback, hours_used, slots_filled = _fit_profile(merged)

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "half_life_weeks": HALF_LIFE_WEEKS,
        "local_timezone": LOCAL_TZ,
        "hours_used": hours_used,
        "slots_filled": slots_filled,
    })
    logger.log_metrics({"baseline": fallback})
    return SeasonalLoadModel(profile=profile.tolist(), fallback=fallback)
