# CrossGuard Dataset Status for Professor Demo

UAV-SEAD access is currently pending on Hugging Face, so the immediate fallback
is PX4 Flight Review public logs. This still gives us real PX4 `.ulg` telemetry
for normal and abnormal flight behavior.

## What We Have Locally

- `data/px4_flight_review/logs/`: good quadrotor logs for normal-flight baseline.
- `data/px4_flight_review/anomaly_logs/`: unsatisfactory quadrotor logs for abnormal examples.
- `data/px4_flight_review/crash_logs/`: crash-rated quadrotor logs for stronger anomaly examples.

## Why This Is Enough for the Meeting

We can show:

1. Real PX4 ULog data is downloaded locally.
2. The logs parse successfully using `pyulog`.
3. The logs contain useful topics for CrossGuard:
   - `battery_status`
   - `vehicle_local_position`
   - `vehicle_global_position`
   - `sensor_gps`
   - `vehicle_acceleration`
   - `vehicle_angular_velocity`
   - `vehicle_attitude`
   - `vehicle_status`
4. We can convert logs into ML-ready feature rows.

## Feature Extraction Command

```powershell
python .\scripts\px4_extract_features.py --limit-per-class 50
```

Output:

```text
data/px4_flight_review/metadata/feature_preview.csv
```

## Message to Professor

UAV-SEAD is still pending access approval, but we are not blocked. We already
have public PX4 Flight Review telemetry locally: normal, unsatisfactory, and
crash-rated quadrotor logs. I added a parser that converts these `.ulg` files
into ML-ready features such as ground speed, vertical speed, acceleration,
battery drop, current draw, voltage, and available sensor topics. This lets us
prototype the adaptive anomaly layer while waiting for UAV-SEAD.
