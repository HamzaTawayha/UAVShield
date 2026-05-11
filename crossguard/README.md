# CrossGuard Sanity Harness

CrossGuard is a runtime sanity checker for agentic UAV systems. It watches
normalized drone state over time and raises alerts when a state claim violates
basic physics, mission logic, or cross-sensor consistency.

## Core Idea

The harness should receive the state before the LLM agent or planner acts on it:

```text
telemetry/perception/depth/commands -> CrossGuard -> validated state + alerts -> planner
```

The current implementation is deterministic and testable without ROS2. The ROS2
adapter in `defense/ros2_node.py` is optional and only runs on a machine with
`rclpy` installed.

## Learned State Sanity Layer

CrossGuard now also supports an optional learned sanity checker. The intent is
to move beyond hand-picked thresholds: UAV-SEAD logs are converted into broad
state windows covering position, speed, acceleration, battery, sensors,
estimator health, actuator outputs, mission state, setpoints, and telemetry
link quality. A MOMENT embedding model plus a lightweight head can then score
whether the whole rolling state sequence looks like real UAV behavior.

The runtime hook is optional:

```python
from crossguard.defense.harness import CrossGuardHarness
from crossguard.ml.moment_checker import MomentStateSanityChecker

ml_checker = MomentStateSanityChecker(
    normalization_path="data/uav_sead/moment_windows/uav_sead_state_windows.normalization.json",
    head_path="data/uav_sead/moment_windows/moment_binary_head/moment_binary_head.joblib",
    threshold_path="data/uav_sead/moment_windows/moment_binary_head/evaluation.json",
)
harness = CrossGuardHarness(ml_checker=ml_checker)
```

When the learned checker flags a window, CrossGuard emits
`ml.state_anomaly` alongside the deterministic invariant violations.

## Implemented Checks

- Battery percentage range and impossible discharge/recharge rates.
- GPS/location jumps that require impossible ground speed.
- Altitude rate limits.
- Velocity versus position-delta consistency.
- Acceleration spikes.
- IMU acceleration versus velocity-delta consistency.
- Roll, pitch, and yaw rate limits.
- Compass/yaw versus direction-of-travel consistency.
- Perception claimed range versus independent depth range.
- False waypoint-reached claims.
- Command-effect sanity for `goto`/`avoid` commands.
- Command plausibility for impossible target distance, deadline speed, climb rate, or yaw rate.
- Sensor freshness and future timestamp checks.
- Packet replay detection when packet IDs are provided.
- Optional geofence checks for state and command targets.
- Rangefinder/ground-distance versus altitude consistency.
- Battery drop versus low current draw consistency.
- Peer-drone consistency for stale peer state, duplicate identities, and false peer waypoint claims.

## Quick Demo

From the repository root:

```powershell
python -m crossguard.demo_replay
```

Run tests:

```powershell
python -m unittest discover -s crossguard/tests
```

Build a small UAV-SEAD window sample:

```powershell
python scripts/uav_sead_build_moment_windows.py --limit 20 --max-normal-windows-per-log 1 --max-anomaly-windows-per-log 2
```

Build the full UAV-SEAD window dataset:

```powershell
python scripts/uav_sead_build_moment_windows.py --include-log-level-anomalies
```

Train the MOMENT head on those windows:

```powershell
python -m pip install momentfm --no-deps
python scripts/train_moment_binary_head.py `
  --windows data/uav_sead/moment_windows/uav_sead_state_windows.npz `
  --output-dir data/uav_sead/moment_windows/moment_binary_head
```

## Streamlit UI

The Streamlit test bench lets you edit a previous state, an injected/current
state, and optional attack fields, then runs the real `CrossGuardHarness` verdict.

Install the UI dependency:

```powershell
python -m pip install -r requirements.txt
```

Launch it from the repository root:

```powershell
streamlit run streamlit_app.py
```

## ROS2 Adapter

The adapter subscribes to standard local simulator topics:

```text
/drone1/telemetry/position
/drone1/telemetry/velocity
/drone1/telemetry/battery
/drone1/imu
/drone1/rangefinder
/drone1/agent/goto_target
```

It publishes:

```text
/crossguard/alert
/crossguard/hover_request
/crossguard/violations
```

Wire `/crossguard/hover_request` into the agent safety override so the drone
hovers when the suspicion score crosses the alert threshold.
