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
