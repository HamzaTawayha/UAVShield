# PX4 Flight Review Starter Dataset

Source: https://docs.px4.io/main/en/dev_log/flight_log_analysis_statistical

The PX4 page points to the public Flight Review log corpus at
https://logs.px4.io/browse and recommends PX4's `download_logs.py` bulk
download script.

## What Was Downloaded

This local starter set contains 10 public `.ulg` logs:

- MAV type: `Quadrotor`
- Rating: `good`
- Latest unique vehicle logs first
- Downloader: `.tools/px4/download_logs.py`
- Destination: `data/px4_flight_review/logs/`

These are a conservative first batch for learning normal flight behavior before
scaling up.

## Files

- `logs/*.ulg`: downloaded PX4 ULog flight files.
- `metadata/manifest.csv`: metadata for the downloaded log IDs.
- `metadata/manifest.json`: full metadata entries from the Flight Review API.
- `metadata/ulog_summary.csv`: parse check and topic summary generated with
  `pyulog`.

## Reproduce Download

From the repository root:

```powershell
python .\.tools\px4\download_logs.py `
  --mav-type Quadrotor `
  --rating good `
  --latest-per-vehicle `
  --max-num 10 `
  --download-folder .\data\px4_flight_review\logs `
  --delay 6 `
  --yes
```

The `--delay 6` argument follows PX4's rate-limit guidance in the downloader.
