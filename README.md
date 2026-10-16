# ode-ablation-load-l0-haiku

## Approach and status

**Problem.** Forecast the net electrical consumption from the grid 24 hours ahead, as hourly mean power in watts. The criterion is in `evaluation.yaml`: RMSE on the hourly mean, threshold 6.1, lower is better. Training history ends at 2026-09-01T00:00:00Z; the test window is 2026-09-01 to 2026-10-01 (UTC).

**Data.** The grid-side meter is the Stromzähler (Iskra MT 175, Tasmota), device `urn:infai:ses:device:8ae74f8f-07c9-42c2-ad1c-701b503adac1`, service "Get Energy Consumption", variable `sensor.MT175.P` (total active power). Its stored history runs from 2024-11-29 to the training end. It was chosen from its name and device type, not from a verified connection-point topology. The unit of the power characteristic (W or kW) was not confirmed. The Qubino three-phase meter (`urn:infai:ses:device:2ac5436e-5538-4eb3-a448-2d77de68e915`) also publishes total power and was the alternative. No weather input is used.

**Operator.**

- `training.py` fits a seasonal profile over the 168 hours of the local week (Europe/Berlin). Each slot is an exponentially recency-weighted mean (half-life 4 weeks) of the hourly means of `sensor.MT175.P` over the 365 days before the training end. The overall weighted mean is also registered as a constant.
- `op.py` issues one forecast per hour, at the first message of hour h, for hour h + 24. The forecast carries the target hour as its result time, which is what the scorer buckets by.
- The committed operator is in constant-reference mode: `infer()` returns the training mean, not the profile. The profile is still fitted and registered. To use it, call `model.predict({"timestamp": target})` without the `"constant"` key.

**Runs** (evaluation window, 648 scored hours, same input and split in each):

| Run | Commit | Forecast | RMSE (W) |
|---|---|---|---|
| `e6d1579e39474febb2c0f79fb214b937` | 64d71db | seasonal profile | 225.59 |
| `5527b2172e46489c97b86560dd0c1f8e` | ab51650 | profile + 24 h level correction | 233.66 |
| `775866099b6f4c6086dca7f35ff982bd` | 9316b07 | constant training mean (186.18 W) | 232.67 |

No run met the threshold of 6.1.

**What this shows.** The hour-of-week profile improves on a constant by about 7 W, roughly 3 percent. The 24-hour mean level correction made the profile worse. Most of the September error is not explained by time of day or day of week.

**Open items.**

- The threshold is about 3 percent of the training mean load. Unless the unit or the target is different from what was assumed, it is unlikely to be reached from this meter's history alone. The criterion is the developer's to keep, change or drop.
- Weather is the likely driver of the remaining error, through heating. A forecast 24 hours ahead needs a weather forecast as an input, not observed weather. The yr.no forecast export is the appropriate source; the open-meteo history export is observed and would overstate skill.
- Not verified: the unit of `sensor.MT175.P`, the placement of the meter at the grid connection point, and how the September error is distributed across the weeks of the month. The weekly breakdown needs the test-window actuals, which the platform does not allow a run to report as metrics after the replay.
- The pod carries Operator Lib 1.8.1 while `pyproject.toml` pins v1.8.2. The replay logic was read in 1.8.1; a run trains against 1.8.2.

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.8.2". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-load-l0-haiku". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-load-l0-haiku:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.8.2", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
