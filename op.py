"""The operator: a 24-hour-ahead hourly forecast of grid net consumption.

MLOperator is the machine-learning half of Operator Lib. It loads the model
registered under this pipeline and operator from MLflow, calls infer() for every
message that matches a selector, and calls train() when there is no model yet or
when need_retraining() says so. Training runs on Ray.

Forecast contract. For every hour h, the operator issues one forecast for hour
h + 24h, at the first message it sees in hour h. The forecast carries the target
hour as its result timestamp, so an evaluation bucketed by result time compares
it with the hourly mean of the target series at that same hour.

Current mode: constant reference. The forecast is the training mean of the target
(the model's overall weighted mean), with no profile and no level correction. This
is the reference that any shape model has to beat, so launch 3 scores it.

Three methods are yours. Everything else belongs to the library.
"""

import datetime
import typing

from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

from training import train_model

UTC = datetime.timezone.utc
HORIZON = datetime.timedelta(hours=24)


class CustomConfig(Config):
    """Deployment configuration, typed.

    The base Config already carries mlflow_url, ray_url and ts_conn. Anything
    added here arrives from the operator's deployment config under the same name,
    with this value as the default.
    """

    # Retrain at most this often, in seconds. A day, so a deployment does not
    # spend its life training.
    retrain_after_s = 86400


def _as_utc_hour(timestamp: datetime.datetime) -> datetime.datetime:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


class Operator(MLOperator):
    configType = CustomConfig

    # The inputs this operator accepts. The forecast does not read the value.
    selectors = [
        Selector({"name": "value", "args": ["value"]}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns, so anything set after it is missing
        # during the replay and overwrites what train() just recorded.
        self.trained_at: typing.Optional[datetime.datetime] = None
        # The hour for which a forecast was last issued. One forecast per hour.
        self.last_issued_hour: typing.Optional[datetime.datetime] = None
        super().init(*args, **kwargs)

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        """Called for every message. Returns (result timestamp, result, new model).

        Issues the constant reference forecast for the hour 24 hours after the
        current one, once per hour. The result timestamp is that target hour.
        """
        if model is None:
            return None, None, None

        hour = _as_utc_hour(timestamp)
        if hour == self.last_issued_hour:
            return None, None, None
        self.last_issued_hour = hour

        target = hour + HORIZON
        prediction = float(model.predict({"timestamp": target, "constant": True}))
        return target, {"prediction": prediction}, None

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        """Called when there is no model, or when need_retraining() said so.

        Runs inside a Ray session the library opened, with an MLflow run already
        started, so params and metrics logged through "logger" land on the run the
        resulting model is registered from.
        """
        self.trained_at = datetime.datetime.now(UTC)
        return train_model(logger)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        """Called after every inference, so it has to be cheap.

        Time-based for now. The profile is refitted once a day from the most recent
        year of history, which is what keeps the seasonal shape current.
        """
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(UTC) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s
