---
language:
- en
license: cc-by-4.0
tags:
- state-estimation-anomaly
- uav-anomaly
- anomaly-detection
- fault-detection
- multivariate-time-series
- uav-anomaly-dataset
- px4-anomaly-dataset
- uav-dataset
- px4-dataset
---


# DATASET CARD for UAV-SEAD

## Citation

**BibTeX:**
```bibtex
@article{kabaoglu2024uavsead,
  title={UAV-SEAD: State Estimation Anomaly Dataset for UAVs},
  author={Kabaoglu, Aykut and Sariel, Sanem},
  journal={arXiv preprint},
  year={2026}
}
```

## DATASET INFORMATION

### Dataset Description
UAV-SEAD (State Estimation Anomaly Dataset) is a comprehensive, large-scale, real-world flight dataset designed to advance research in Fault Detection and Identification (FDI) for Unmanned Aerial Vehicles. It contains 1,396 categorized real-world flight logs totaling over 52.4 hours of flight data. Unlike existing benchmarks, it avoids synthetic fault injection, providing naturally occurring anomalies across diverse environmental and hardware configurations.

- **Curated by:** Aykut Kabaoglu
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### Dataset Sources
- **Paper:** "UAV-SEAD: State Estimation Anomaly Dataset for UAVs"

### Toolbox for Dataset Preprocessing and Annotation
- https://github.com/aykutkabaoglu/ulog_annotation_tool
- Use readme in the `ulog_annotation_tool` repository to start using dataset. The first step is converting raw flight logs into csv files. Then the flight logs can be used for plotting and labeling.

## DATASET CREATION AND USAGE

### Dataset Structure

The dataset is structured as a collection of multivariate time-series sequences. 
- **Point Anomalies:** Spikes and outliers in sensor readings.
- **Contextual Anomalies:** Readings that are only anomalous relative to the flight state (e.g., zero altitude during thrust).
- **Collective Anomalies:** Sustained drifts or patterns indicating gradual system failure.

### Use Cases
- Benchmarking multivariate time-series anomaly detection algorithms.
- Developing predictive maintenance and fault identification systems for quadrotors.

#### Data Collection and Processing
Data was harvested from a fleet of custom-built, PX4-based quadrotors. 
- **Hardware:** Multiple airframes with variations in motor sizes, weights, and sensor suites. Check paper for hardware details.
- **Environments:** Flight missions conducted in indoor hangars, warehouses, and open outdoor spaces.

#### Annotation process
Annotation was performed via **Physically-Constrained Expert Evaluation**. Human experts identified state estimation divergences by comparing autopilot estimates with real-world observed outcomes. Labels were verified against physical flight dynamics to ensure the statistical anomalies correlate with actual physical events.

### Bias, Risks, and Limitations of Data Collection
- **Platform Specificity:** Limited to multi-rotor dynamics (quadrotors). Limited outdoor flights with GPS availability.


## DATASET STATISTICS

### Annotated Data Flight Source Statistics
Annotated Number of Flight: 1396

| Type | Time (hh:mm:ss)} |
|--------|----:|
| Total flight duration | 52:20:05 |
| External position data duration | 51:28:34 |
| Global position data duration | 04:35:24 |
| Distance sensor data duration | 51:35:05 |
| Barometer altitude data duration | 26:23:45 |

### Annotated Data Class Statistics

| Class Labeling | # Flights | Total Flight Duration (hh:mm:ss)} | Anomaly Duration (hh:mm:ss) |
|--------|----|---------:|----------:|
| Normal | 900 | 38:24:19 |  -- | -- | -- |
| Mechanical and Electrical | 47 | 02:12:38 | 00:12:43 |
| External Position | 197 | 05:12:25 | 01:17:02 |
| Global Position | 41 | 01:40:42 | 00:36:23 |
| Altitude | 78 | 02:41:21 | 00:25:57 |
| Uncategorized | 141 | 03:13:09 | -- |


### RAW Data Statistics
Number of Flights: 3196

### RAW Data, Data Fields
The dataset follows a **Superset Schema** of 1897 unique columns. Fields are grouped into 84 unique topics by PX4 UORB topic prefixes.

81 PX4 Topics out of 84 and their fields that has data at least a single flight log in the dataset are listed as below:

| # | Topic (Superset) | # Active Subfields | Present in # Files | Subfields |
|---|------------------|-------------------:|--------------------:|-----------|
| 1 | **action_request** | 3 | 5 (0.2%) | `action`, `mode`, `source` |
| 2 | **actuator_armed** | 5 | 8 (0.3%) | `armed`, `armed_time_ms`, `manual_lockdown`, `prearmed`, `ready_to_arm` |
| 3 | **actuator_controls_0** | 6 | 3196 (100.0%) | `control[0]`, `control[1]`, `control[2]`, `control[3]`, `control[7]`, `timestamp_sample` |
| 4 | **actuator_controls_3** | 4 | 5 (0.2%) | `control[0]`, `control[1]`, `control[2]`, `control[3]` |
| 5 | **actuator_outputs** | 9 | 3191 (99.8%) | `noutputs`, `output[0]`, `output[1]`, `output[2]`, `output[3]`, `output[4]`, `output[5]`, `output[6]`, `output[7]` |
| 6 | **battery_status** | 13 | 3196 (100.0%) | `cell_count`, `connected`, `current_a`, `current_average_a`, `current_filtered_a`, `discharged_mah`, `id`, `remaining`, `scale`, `system_source`, `voltage_filtered_v`, `voltage_v`, `warning` |
| 7 | **camera_trigger** | 2 | 389 (12.2%) | `seq`, `timestamp_utc` |
| 8 | **commander_state** | 2 | 8 (0.3%) | `main_state`, `main_state_changes` |
| 9 | **cpuload** | 2 | 3196 (100.0%) | `load`, `ram_usage` |
| 10 | **distance_sensor** | 9 | 3118 (97.6%) | `current_distance`, `device_id`, `h_fov`, `id`, `max_distance`, `min_distance`, `orientation`, `signal_quality`, `v_fov` |
| 11 | **ekf2_innovations** | 29 | 3169 (99.2%) | `flow_innov[0]`, `flow_innov[1]`, `flow_innov_var[0]`, `flow_innov_var[1]`, `hagl_innov`, `hagl_innov_var`, `heading_innov`, `heading_innov_var`, `mag_innov[0]`, `mag_innov[1]`, `mag_innov[2]`, `mag_innov_var[0]`, `mag_innov_var[1]`, `mag_innov_var[2]`, `output_tracking_error[0]`, `output_tracking_error[1]`, `output_tracking_error[2]`, `vel_pos_innov[0]`, `vel_pos_innov[1]`, `vel_pos_innov[2]`, `vel_pos_innov[3]`, `vel_pos_innov[4]`, `vel_pos_innov[5]`, `vel_pos_innov_var[0]`, `vel_pos_innov_var[1]`, `vel_pos_innov_var[2]`, `vel_pos_innov_var[3]`, `vel_pos_innov_var[4]`, `vel_pos_innov_var[5]` |
| 12 | **ekf2_timestamps** | 9 | 1098 (34.4%) | `airspeed_timestamp_rel`, `distance_sensor_timestamp_rel`, `gps_timestamp_rel`, `optical_flow_timestamp_rel`, `vehicle_air_data_timestamp_rel`, `vehicle_magnetometer_timestamp_rel`, `vision_attitude_timestamp_rel`, `vision_position_timestamp_rel`, `visual_odometry_timestamp_rel` |
| 13 | **estimator_attitude** | 10 | 5 (0.2%) | `delta_q_reset[0]`, `delta_q_reset[1]`, `delta_q_reset[2]`, `delta_q_reset[3]`, `q[0]`, `q[1]`, `q[2]`, `q[3]`, `quat_reset_counter`, `timestamp_sample` |
| 14 | **estimator_event_flags** | 7 | 5 (0.2%) | `height_sensor_timeout`, `information_event_changes`, `reset_pos_to_vision`, `starting_vision_pos_fusion`, `starting_vision_yaw_fusion`, `timestamp_sample`, `warning_event_changes` |
| 15 | **estimator_innovation_test_ratios** | 13 | 8 (0.3%) | `baro_vpos`, `ev_hpos[0]`, `ev_hpos[1]`, `ev_hvel[1]`, `ev_vpos`, `flow[1]`, `gps_hpos[0]`, `gps_hpos[1]`, `gps_hvel[1]`, `hagl`, `hagl_rate`, `heading`, `timestamp_sample` |
| 16 | **estimator_innovation_variances** | 10 | 8 (0.3%) | `baro_vpos`, `ev_hpos[0]`, `ev_hpos[1]`, `ev_vpos`, `gps_hpos[0]`, `gps_hpos[1]`, `hagl`, `hagl_rate`, `heading`, `timestamp_sample` |
| 17 | **estimator_innovations** | 10 | 8 (0.3%) | `baro_vpos`, `ev_hpos[0]`, `ev_hpos[1]`, `ev_vpos`, `gps_hpos[0]`, `gps_hpos[1]`, `hagl`, `hagl_rate`, `heading`, `timestamp_sample` |
| 18 | **estimator_local_position** | 36 | 5 (0.2%) | `ax`, `ay`, `az`, `delta_heading`, `delta_vxy[0]`, `delta_vxy[1]`, `delta_vz`, `delta_xy[0]`, `delta_xy[1]`, `delta_z`, `dist_bottom`, `dist_bottom_sensor_bitfield`, `dist_bottom_valid`, `eph`, `epv`, `evh`, `evv`, `heading`, `heading_good_for_control`, `heading_reset_counter`, `timestamp_sample`, `v_xy_valid`, `v_z_valid`, `vx`, `vxy_reset_counter`, `vy`, `vz`, `vz_reset_counter`, `x`, `xy_reset_counter`, `xy_valid`, `y`, `z`, `z_deriv`, `z_reset_counter`, `z_valid` |
| 19 | **estimator_selector_status** | 14 | 5 (0.2%) | `accel_device_id`, `baro_device_id`, `combined_test_ratio[0]`, `combined_test_ratio[1]`, `gyro_device_id`, `healthy[0]`, `healthy[1]`, `instance_changed_count`, `instances_available`, `last_instance_change`, `mag_device_id`, `primary_instance`, `relative_test_ratio[0]`, `relative_test_ratio[1]` |
| 20 | **estimator_sensor_bias** | 27 | 8 (0.3%) | `accel_bias[0]`, `accel_bias[1]`, `accel_bias[2]`, `accel_bias_limit`, `accel_bias_stable`, `accel_bias_valid`, `accel_bias_variance[0]`, `accel_bias_variance[1]`, `accel_bias_variance[2]`, `accel_device_id`, `gyro_bias[0]`, `gyro_bias[1]`, `gyro_bias[2]`, `gyro_bias_limit`, `gyro_bias_stable`, `gyro_bias_valid`, `gyro_bias_variance[0]`, `gyro_bias_variance[1]`, `gyro_bias_variance[2]`, `gyro_device_id`, `mag_bias_limit`, `mag_bias_valid`, `mag_bias_variance[0]`, `mag_bias_variance[1]`, `mag_bias_variance[2]`, `mag_device_id`, `timestamp_sample` |
| 21 | **estimator_states** | 37 | 5 (0.2%) | `covariances[0]`, `covariances[10]`, `covariances[11]`, `covariances[12]`, `covariances[13]`, `covariances[14]`, `covariances[15]`, `covariances[1]`, `covariances[2]`, `covariances[3]`, `covariances[4]`, `covariances[5]`, `covariances[6]`, `covariances[7]`, `covariances[8]`, `covariances[9]`, `n_states`, `states[0]`, `states[10]`, `states[11]`, `states[12]`, `states[13]`, `states[14]`, `states[15]`, `states[16]`, `states[17]`, `states[18]`, `states[1]`, `states[2]`, `states[3]`, `states[4]`, `states[5]`, `states[6]`, `states[7]`, `states[8]`, `states[9]`, `timestamp_sample` |
| 22 | **estimator_status** | 81 | 3194 (99.9%) | `accel_device_id`, `baro_device_id`, `control_mode_flags`, `covariances[0]`, `covariances[10]`, `covariances[11]`, `covariances[12]`, `covariances[13]`, `covariances[14]`, `covariances[15]`, `covariances[16]`, `covariances[17]`, `covariances[18]`, `covariances[19]`, `covariances[1]`, `covariances[20]`, `covariances[21]`, `covariances[2]`, `covariances[3]`, `covariances[4]`, `covariances[5]`, `covariances[6]`, `covariances[7]`, `covariances[8]`, `covariances[9]`, `filter_fault_flags`, `gps_check_fail_flags`, `gyro_device_id`, `hagl_test_ratio`, `health_flags`, `hgt_test_ratio`, `innovation_check_flags`, `mag_device_id`, `mag_test_ratio`, `n_states`, `nan_flags`, `output_tracking_error[0]`, `output_tracking_error[1]`, `output_tracking_error[2]`, `pos_horiz_accuracy`, `pos_test_ratio`, `pos_vert_accuracy`, `pre_flt_fail`, `pre_flt_fail_innov_heading`, `pre_flt_fail_innov_height`, `pre_flt_fail_mag_field_disturbed`, `reset_count_pod_d`, `reset_count_pos_ne`, `reset_count_quat`, `reset_count_vel_d`, `reset_count_vel_ne`, `solution_status_flags`, `states[0]`, `states[10]`, `states[11]`, `states[12]`, `states[13]`, `states[14]`, `states[15]`, `states[16]`, `states[17]`, `states[18]`, `states[19]`, `states[1]`, `states[20]`, `states[21]`, `states[2]`, `states[3]`, `states[4]`, `states[5]`, `states[6]`, `states[7]`, `states[8]`, `states[9]`, `time_slip`, `timeout_flags`, `timestamp_sample`, `vel_test_ratio`, `vibe[0]`, `vibe[1]`, `vibe[2]` |
| 23 | **estimator_status_flags** | 16 | 5 (0.2%) | `control_status_changes`, `cs_ev_hgt`, `cs_ev_pos`, `cs_ev_yaw`, `cs_gnd_effect`, `cs_in_air`, `cs_inertial_dead_reckoning`, `cs_mag_field_disturbed`, `cs_rng_kin_consistent`, `cs_tilt_align`, `cs_vehicle_at_rest`, `cs_yaw_align`, `innovation_fault_status_changes`, `reject_hagl`, `reject_ver_pos`, `timestamp_sample` |
| 24 | **estimator_visual_odometry_aligned** | 9 | 5 (0.2%) | `q[0]`, `q[1]`, `q[2]`, `q[3]`, `q_offset[0]`, `timestamp_sample`, `x`, `y`, `z` |
| 25 | **event** | 10 | 5 (0.2%) | `arguments[0]`, `arguments[1]`, `arguments[2]`, `arguments[3]`, `arguments[4]`, `arguments[5]`, `arguments[6]`, `event_sequence`, `id`, `log_levels` |
| 26 | **failure_detector_status** | 1 | 5 (0.2%) | `imbalanced_prop_metric` |
| 27 | **home_position** | 11 | 104 (3.3%) | `alt`, `lat`, `lon`, `manual_home`, `valid_alt`, `valid_hpos`, `valid_lpos`, `x`, `y`, `yaw`, `z` |
| 28 | **hover_thrust_estimate** | 8 | 5 (0.2%) | `accel_innov`, `accel_innov_test_ratio`, `accel_innov_var`, `accel_noise_var`, `hover_thrust`, `hover_thrust_var`, `timestamp_sample`, `valid` |
| 29 | **input_rc** | 26 | 3195 (100.0%) | `channel_count`, `input_source`, `rc_failsafe`, `rc_lost_frame_count`, `rc_ppm_frame_length`, `rc_total_frame_count`, `rssi`, `timestamp_last_signal`, `values[0]`, `values[10]`, `values[11]`, `values[12]`, `values[13]`, `values[14]`, `values[15]`, `values[16]`, `values[17]`, `values[1]`, `values[2]`, `values[3]`, `values[4]`, `values[5]`, `values[6]`, `values[7]`, `values[8]`, `values[9]` |
| 30 | **logger_status** | 8 | 3 (0.1%) | `backend`, `buffer_size_bytes`, `buffer_used_bytes`, `dropouts`, `message_gaps`, `num_messages`, `total_written_kb`, `write_rate_kb_s` |
| 31 | **magnetometer_bias_estimate** | 3 | 5 (0.2%) | `bias_x[0]`, `bias_y[0]`, `bias_z[0]` |
| 32 | **manual_control_setpoint** | 11 | 3194 (99.9%) | `aux1`, `data_source`, `kill_switch`, `mode_slot`, `r`, `sticks_moving`, `timestamp_sample`, `valid`, `x`, `y`, `z` |
| 33 | **manual_control_switches** | 4 | 5 (0.2%) | `kill_switch`, `mode_slot`, `switch_changes`, `timestamp_sample` |
| 34 | **mission_result** | 4 | 2150 (67.3%) | `finished`, `instance_count`, `seq_reached`, `valid` |
| 35 | **multirotor_motor_limits** | 1 | 3 (0.1%) | `saturation_status` |
| 36 | **offboard_control_mode** | 6 | 8 (0.3%) | `ignore_acceleration_force`, `ignore_bodyrate_x`, `ignore_bodyrate_y`, `ignore_bodyrate_z`, `ignore_velocity`, `position` |
| 37 | **optical_flow** | 14 | 991 (31.0%) | `frame_count_since_last_readout`, `ground_distance_m`, `gyro_temperature`, `gyro_x_rate_integral`, `gyro_y_rate_integral`, `gyro_z_rate_integral`, `integration_timespan`, `max_flow_rate`, `max_ground_distance`, `min_ground_distance`, `pixel_flow_x_integral`, `pixel_flow_y_integral`, `quality`, `time_since_last_sonar_update` |
| 38 | **parameter_update** | 8 | 4 (0.1%) | `active`, `changed`, `custom_default`, `export_count`, `find_count`, `get_count`, `instance`, `set_count` |
| 39 | **ping** | 5 | 1627 (50.9%) | `dropped_packets`, `ping_sequence`, `ping_time`, `rtt_ms`, `system_id` |
| 40 | **position_setpoint_triplet** | 35 | 2574 (80.5%) | `current.acceptance_radius`, `current.alt`, `current.alt_valid`, `current.cruising_speed`, `current.cruising_throttle`, `current.lat`, `current.loiter_direction`, `current.loiter_radius`, `current.lon`, `current.position_valid`, `current.timestamp`, `current.type`, `current.valid`, `current.velocity_frame`, `current.velocity_valid`, `current.vx`, `current.vy`, `current.vz`, `current.x`, `current.y`, `current.yaw`, `current.yaw_valid`, `current.z`, `next.acceptance_radius`, `next.cruising_speed`, `next.cruising_throttle`, `next.loiter_radius`, `next.timestamp`, `next.type`, `previous.acceptance_radius`, `previous.cruising_speed`, `previous.cruising_throttle`, `previous.loiter_radius`, `previous.timestamp`, `previous.type` |
| 41 | **rate_ctrl_status** | 7 | 3129 (97.9%) | `additional_integ1`, `pitchspeed`, `pitchspeed_integ`, `rollspeed`, `rollspeed_integ`, `yawspeed`, `yawspeed_integ` |
| 42 | **safety** | 2 | 1728 (54.1%) | `safety_off`, `safety_switch_available` |
| 43 | **sensor_accel** | 17 | 520 (16.3%) | `device_id`, `integral_dt`, `range_m_s2`, `samples`, `scaling`, `temperature`, `temperature_raw`, `timestamp_sample`, `x`, `x_integral`, `x_raw`, `y`, `y_integral`, `y_raw`, `z`, `z_integral`, `z_raw` |
| 44 | **sensor_baro** | 5 | 539 (16.9%) | `altitude`, `device_id`, `pressure`, `temperature`, `timestamp_sample` |
| 45 | **sensor_combined** | 17 | 3196 (100.0%) | `accel_calibration_count`, `accelerometer_integral_dt`, `accelerometer_m_s2[0]`, `accelerometer_m_s2[1]`, `accelerometer_m_s2[2]`, `baro_alt_meter`, `baro_temp_celcius`, `baro_timestamp_relative`, `gyro_calibration_count`, `gyro_integral_dt`, `gyro_rad[0]`, `gyro_rad[1]`, `gyro_rad[2]`, `magnetometer_ga[0]`, `magnetometer_ga[1]`, `magnetometer_ga[2]`, `magnetometer_timestamp_relative` |
| 46 | **sensor_gyro** | 17 | 520 (16.3%) | `device_id`, `integral_dt`, `range_rad_s`, `samples`, `scaling`, `temperature`, `temperature_raw`, `timestamp_sample`, `x`, `x_integral`, `x_raw`, `y`, `y_integral`, `y_raw`, `z`, `z_integral`, `z_raw` |
| 47 | **sensor_gyro_fft** | 22 | 5 (0.2%) | `device_id`, `peak_frequencies_x[0]`, `peak_frequencies_x[1]`, `peak_frequencies_x[2]`, `peak_frequencies_y[0]`, `peak_frequencies_y[1]`, `peak_frequencies_y[2]`, `peak_frequencies_z[0]`, `peak_frequencies_z[1]`, `peak_frequencies_z[2]`, `peak_snr_x[0]`, `peak_snr_x[1]`, `peak_snr_x[2]`, `peak_snr_y[0]`, `peak_snr_y[1]`, `peak_snr_y[2]`, `peak_snr_z[0]`, `peak_snr_z[1]`, `peak_snr_z[2]`, `resolution_hz`, `sensor_sample_rate_hz`, `timestamp_sample` |
| 48 | **sensor_mag** | 13 | 520 (16.3%) | `device_id`, `error_count`, `is_external`, `range_ga`, `scaling`, `temperature`, `timestamp_sample`, `x`, `x_raw`, `y`, `y_raw`, `z`, `z_raw` |
| 49 | **sensor_preflight** | 3 | 3188 (99.7%) | `accel_inconsistency_m_s_s`, `gyro_inconsistency_rad_s`, `mag_inconsistency_ga` |
| 50 | **sensor_selection** | 4 | 500 (15.6%) | `accel_device_id`, `baro_device_id`, `gyro_device_id`, `mag_device_id` |
| 51 | **sensors_status_imu** | 18 | 5 (0.2%) | `accel_device_id_primary`, `accel_device_ids[0]`, `accel_device_ids[1]`, `accel_healthy[0]`, `accel_healthy[1]`, `accel_inconsistency_m_s_s[0]`, `accel_inconsistency_m_s_s[1]`, `accel_priority[0]`, `accel_priority[1]`, `gyro_device_id_primary`, `gyro_device_ids[0]`, `gyro_device_ids[1]`, `gyro_healthy[0]`, `gyro_healthy[1]`, `gyro_inconsistency_rad_s[0]`, `gyro_inconsistency_rad_s[1]`, `gyro_priority[0]`, `gyro_priority[1]` |
| 52 | **system_power** | 10 | 3196 (100.0%) | `brick_valid`, `sensors3v3[0]`, `sensors3v3_valid`, `servo_valid`, `usb_connected`, `usb_valid`, `v3v3_valid`, `voltage3v3_v`, `voltage5V_v`, `voltage5v_v` |
| 53 | **takeoff_status** | 2 | 5 (0.2%) | `takeoff_state`, `tilt_limit` |
| 54 | **telemetry_status** | 45 | 3174 (99.3%) | `component_id`, `data_rate`, `flow_control`, `forwarding`, `ftp`, `heartbeat_component_udp_bridge`, `heartbeat_time`, `heartbeat_type_gcs`, `heartbeat_type_onboard_controller`, `heartbeats[0].state`, `heartbeats[0].system_id`, `heartbeats[0].timestamp`, `heartbeats[0].type`, `heartbeats[1].component_id`, `heartbeats[1].state`, `heartbeats[1].system_id`, `heartbeats[1].timestamp`, `heartbeats[1].type`, `mavlink_v2`, `mode`, `noise`, `rate_multiplier`, `rate_rx`, `rate_tx`, `rate_txerr`, `remote_component_id`, `remote_noise`, `remote_rssi`, `remote_system_id`, `remote_system_status`, `remote_type`, `rssi`, `rx_message_count`, `rx_message_lost_count`, `rx_message_lost_rate`, `rx_rate_avg`, `rxerrors`, `streams`, `system_id`, `telem_time`, `tx_buffer_overruns`, `tx_message_count`, `tx_rate_avg`, `txbuf`, `type` |
| 55 | **timestamp** | 0 | 3196 (100.0%) | *(standalone field)* |
| 56 | **timesync_status** | 4 | 1698 (53.1%) | `estimated_offset`, `observed_offset`, `remote_timestamp`, `round_trip_time` |
| 57 | **trajectory_setpoint** | 22 | 1126 (35.2%) | `acc_x`, `acc_y`, `acc_z`, `acceleration[0]`, `acceleration[1]`, `acceleration[2]`, `jerk[0]`, `jerk[1]`, `jerk[2]`, `jerk_x`, `jerk_y`, `jerk_z`, `thrust[0]`, `thrust[1]`, `vx`, `vy`, `vz`, `x`, `y`, `yaw`, `yawspeed`, `z` |
| 58 | **vehicle_acceleration** | 4 | 5 (0.2%) | `timestamp_sample`, `xyz[0]`, `xyz[1]`, `xyz[2]` |
| 59 | **vehicle_air_data** | 6 | 2005 (62.7%) | `baro_alt_meter`, `baro_device_id`, `baro_pressure_pa`, `baro_temp_celcius`, `rho`, `timestamp_sample` |
| 60 | **vehicle_angular_acceleration** | 4 | 8 (0.3%) | `timestamp_sample`, `xyz[0]`, `xyz[1]`, `xyz[2]` |
| 61 | **vehicle_angular_velocity** | 4 | 1409 (44.1%) | `timestamp_sample`, `xyz[0]`, `xyz[1]`, `xyz[2]` |
| 62 | **vehicle_attitude** | 13 | 3179 (99.5%) | `delta_q_reset[0]`, `delta_q_reset[1]`, `delta_q_reset[2]`, `delta_q_reset[3]`, `pitchspeed`, `q[0]`, `q[1]`, `q[2]`, `q[3]`, `quat_reset_counter`, `rollspeed`, `timestamp_sample`, `yawspeed` |
| 63 | **vehicle_attitude_setpoint** | 12 | 3196 (100.0%) | `landing_gear`, `pitch_body`, `q_d[0]`, `q_d[1]`, `q_d[2]`, `q_d[3]`, `q_d_valid`, `roll_body`, `thrust`, `thrust_body[2]`, `yaw_body`, `yaw_sp_move_rate` |
| 64 | **vehicle_command** | 14 | 2782 (87.0%) | `command`, `confirmation`, `from_external`, `param1`, `param2`, `param3`, `param4`, `param5`, `param6`, `param7`, `source_component`, `source_system`, `target_component`, `target_system` |
| 65 | **vehicle_constraints** | 2 | 2 (0.1%) | `speed_down`, `speed_up` |
| 66 | **vehicle_control_mode** | 12 | 8 (0.3%) | `flag_armed`, `flag_control_acceleration_enabled`, `flag_control_altitude_enabled`, `flag_control_attitude_enabled`, `flag_control_auto_enabled`, `flag_control_climb_rate_enabled`, `flag_control_manual_enabled`, `flag_control_offboard_enabled`, `flag_control_position_enabled`, `flag_control_rates_enabled`, `flag_control_velocity_enabled`, `flag_multicopter_position_control_enabled` |
| 67 | **vehicle_global_position** | 14 | 102 (3.2%) | `alt`, `dead_reckoning`, `delta_alt`, `eph`, `epv`, `lat`, `lat_lon_reset_counter`, `lon`, `terrain_alt`, `terrain_alt_valid`, `vel_d`, `vel_e`, `vel_n`, `yaw` |
| 68 | **vehicle_gps_position** | 21 | 184 (5.8%) | `alt`, `alt_ellipsoid`, `c_variance_rad`, `cog_rad`, `eph`, `epv`, `fix_type`, `hdop`, `jamming_indicator`, `lat`, `lon`, `noise_per_ms`, `s_variance_m_s`, `satellites_used`, `time_utc_usec`, `vdop`, `vel_d_m_s`, `vel_e_m_s`, `vel_m_s`, `vel_n_m_s`, `vel_ned_valid` |
| 69 | **vehicle_imu** | 13 | 8 (0.3%) | `accel_calibration_count`, `accel_device_id`, `delta_angle[0]`, `delta_angle[1]`, `delta_angle[2]`, `delta_angle_dt`, `delta_velocity[0]`, `delta_velocity[1]`, `delta_velocity[2]`, `delta_velocity_dt`, `gyro_calibration_count`, `gyro_device_id`, `timestamp_sample` |
| 70 | **vehicle_imu_status** | 24 | 8 (0.3%) | `accel_device_id`, `accel_rate_hz`, `accel_raw_rate_hz`, `accel_vibration_metric`, `delta_angle_coning_metric`, `gyro_coning_vibration`, `gyro_device_id`, `gyro_rate_hz`, `gyro_raw_rate_hz`, `gyro_vibration_metric`, `mean_accel[0]`, `mean_accel[1]`, `mean_accel[2]`, `mean_gyro[0]`, `mean_gyro[1]`, `mean_gyro[2]`, `temperature_accel`, `temperature_gyro`, `var_accel[0]`, `var_accel[1]`, `var_accel[2]`, `var_gyro[0]`, `var_gyro[1]`, `var_gyro[2]` |
| 71 | **vehicle_land_detected** | 12 | 3196 (100.0%) | `alt_max`, `at_rest`, `close_to_ground_or_skipped_check`, `freefall`, `ground_contact`, `has_low_throttle`, `horizontal_movement`, `in_descend`, `in_ground_effect`, `landed`, `maybe_landed`, `vertical_movement` |
| 72 | **vehicle_local_position** | 47 | 3177 (99.4%) | `ax`, `ay`, `az`, `delta_heading`, `delta_vxy[0]`, `delta_vxy[1]`, `delta_vz`, `delta_xy[0]`, `delta_xy[1]`, `delta_z`, `dist_bottom`, `dist_bottom_rate`, `dist_bottom_sensor_bitfield`, `dist_bottom_valid`, `eph`, `epv`, `evh`, `evv`, `hagl_max`, `hagl_min`, `heading`, `heading_good_for_control`, `heading_reset_counter`, `ref_alt`, `ref_lat`, `ref_lon`, `ref_timestamp`, `timestamp_sample`, `v_xy_valid`, `v_z_valid`, `vx`, `vxy_max`, `vxy_reset_counter`, `vy`, `vz`, `vz_reset_counter`, `x`, `xy_global`, `xy_reset_counter`, `xy_valid`, `y`, `yaw`, `z`, `z_deriv`, `z_global`, `z_reset_counter`, `z_valid` |
| 73 | **vehicle_local_position_setpoint** | 17 | 2490 (77.9%) | `acc_x`, `acc_y`, `acc_z`, `acceleration[0]`, `acceleration[1]`, `acceleration[2]`, `thrust[0]`, `thrust[1]`, `thrust[2]`, `vx`, `vy`, `vz`, `x`, `y`, `yaw`, `yawspeed`, `z` |
| 74 | **vehicle_magnetometer** | 6 | 2005 (62.7%) | `calibration_count`, `device_id`, `magnetometer_ga[0]`, `magnetometer_ga[1]`, `magnetometer_ga[2]`, `timestamp_sample` |
| 75 | **vehicle_rates_setpoint** | 5 | 3196 (100.0%) | `pitch`, `roll`, `thrust`, `thrust_body[2]`, `yaw` |
| 76 | **vehicle_status** | 20 | 3196 (100.0%) | `armed_time`, `arming_state`, `component_id`, `data_link_lost`, `data_link_lost_counter`, `failsafe`, `failure_detector_status`, `is_rotary_wing`, `latest_arming_reason`, `latest_disarming_reason`, `nav_state`, `nav_state_timestamp`, `onboard_control_sensors_enabled`, `onboard_control_sensors_health`, `onboard_control_sensors_present`, `rc_signal_lost`, `system_id`, `system_type`, `takeoff_time`, `vehicle_type` |
| 77 | **vehicle_status_flags** | 35 | 3185 (99.7%) | `angular_velocity_valid`, `attitude_valid`, `battery_healthy`, `circuit_breaker_engaged_enginefailure_check`, `circuit_breaker_engaged_gpsfailure_check`, `circuit_breaker_engaged_usb_check`, `circuit_breaker_flight_termination_disabled`, `circuit_breakers`, `condition_auto_mission_available`, `condition_battery_healthy`, `condition_global_position_valid`, `condition_home_position_valid`, `condition_local_altitude_valid`, `condition_local_position_valid`, `condition_local_velocity_valid`, `condition_power_input_valid`, `condition_system_hotplug_timeout`, `condition_system_sensors_initialized`, `conditions`, `dead_reckoning`, `flight_terminated`, `local_altitude_valid`, `local_position_valid`, `local_velocity_valid`, `offboard_control_loss_timeout`, `offboard_control_signal_lost`, `other_flags`, `position_reliant_on_vision_position`, `power_input_valid`, `rc_calibration_valid`, `rc_signal_found_once`, `sd_card_detected_once`, `system_hotplug_timeout`, `system_sensors_initialized`, `usb_connected` |
| 78 | **vehicle_vision_attitude** | 4 | 1740 (54.4%) | `q[0]`, `q[1]`, `q[2]`, `q[3]` |
| 79 | **vehicle_vision_position** | 7 | 1740 (54.4%) | `v_xy_valid`, `v_z_valid`, `x`, `xy_valid`, `y`, `z`, `z_valid` |
| 80 | **vehicle_visual_odometry** | 20 | 521 (16.3%) | `pose_covariance[0]`, `pose_covariance[11]`, `pose_covariance[15]`, `pose_covariance[17]`, `pose_covariance[18]`, `pose_covariance[19]`, `pose_covariance[1]`, `pose_covariance[20]`, `pose_covariance[2]`, `pose_covariance[6]`, `pose_covariance[7]`, `q[0]`, `q[1]`, `q[2]`, `q[3]`, `timestamp_sample`, `velocity_frame`, `x`, `y`, `z` |
| 81 | **vehicle_visual_odometry_aligned** | 19 | 516 (16.1%) | `pose_covariance[0]`, `pose_covariance[11]`, `pose_covariance[15]`, `pose_covariance[17]`, `pose_covariance[18]`, `pose_covariance[19]`, `pose_covariance[1]`, `pose_covariance[20]`, `pose_covariance[2]`, `pose_covariance[6]`, `pose_covariance[7]`, `q[0]`, `q[1]`, `q[2]`, `q[3]`, `q_offset[0]`, `x`, `y`, `z` |



###  Dataset Features
Major PX4 topics that can mostly be found in flight logs and their descriptions:

| Group # | Prefix | Count | Description / Key Examples |
| :--- | :--- | :---: | :--- |
| 1 | `actuator_controls_0` | 9 | Pilot control inputs (roll, pitch, yaw, thrust). |
| 2 | `actuator_outputs` | 17 | Final PWM motor outputs and servo positions. |
| 3 | `battery_status` | 25 | Power metrics including voltage, current, and cell data. |
| 4 | `camera_trigger` | 3 | Image capture events and UTC timestamps. |
| 5 | `cpuload` | 2 | Flight controller CPU and RAM utilization. |
| 6 | `distance_sensor` | 15 | Lidar/Ultrasound range data and variance. |
| 7 | `ekf2_innovations` | 39 | Kalman filter health and sensor discrepancies. |
| 8 | `ekf2_timestamps` | 9 | Relative timestamps for specific sensor updates in EKF. |
| 9 | `estimator_status` | 75 | Internal EKF states, covariances, and health flags. |
| 10 | `input_rc` | 27 | Raw Remote Control (RC) input values and signal strength. |
| 11 | `manual_control_setpoint` | 26 | Manual stick positions and hardware switch states. |
| 12 | `mission_result` | 14 | Mission progress, sequence reached, and failure flags. |
| 37 | `optical_flow` | 14 | Ground velocity estimations and surface quality from optical flow sensors. |
| 13 | `ping` | 6 | Latency and round-trip time for system communication. |
| 14 | `position_setpoint_triplet` | 123 | Target coordinates for previous, current, and next waypoints. |
| 15 | `rate_ctrl_status` | 7 | Internal states of the angular rate controller. |
| 16 | `safety` | 4 | Safety switch status and override availability. |
| 17 | `sensor_accel` | 16 | Raw and filtered accelerometer data in $m/s^2$. |
| 18 | `sensor_baro` | 4 | Static pressure and temperature from the barometer. |
| 19 | `sensor_combined` | 9 | Synchronized and downsampled IMU data. |
| 20 | `sensor_gyro` | 16 | Raw and filtered gyroscope data in $rad/s$. |
| 21 | `sensor_mag` | 12 | Magnetometer (compass) field strength and device info. |
| 22 | `sensor_preflight` | 4 | Pre-arm sensor inconsistency and calibration checks. |
| 23 | `sensor_selection` | 4 | Device IDs for the currently active/selected sensors. |
| 24 | `system_power` | 13 | Internal rail voltages (5V, 3.3V) and hardware status. |
| 25 | `telemetry_status` | 27 | MAVLink communication health and data rates. |
| 26 | `timestamp` | 1 | Master log entry time (Microseconds since boot). |
| 27 | `timesync_status` | 4 | Clock synchronization offsets and round-trip time. |
| 28 | `trajectory_setpoint` | 17 | Targeted position, velocity, and acceleration vectors. |
| 29 | `vehicle_air_data` | 4 | Barometric altitude and air density calculations. |
| 30 | `vehicle_angular_velocity` | 4 | Compensated rotational rates of the vehicle body. |
| 31 | `vehicle_attitude` | 12 | Current orientation (Quaternions) and Euler rates. |
| 32 | `vehicle_attitude_setpoint` | 20 | Desired orientation and thrust for the flight controller. |
| 33 | `vehicle_command` | 14 | Logged MAVLink commands and execution parameters. |
| 67 | `vehicle_global_position` | 14 | Fused global position data including latitude, longitude, and altitude above MSL. |
| 68 | `vehicle_gps_position` | 21 | GPS coordinates (lat, lon, alt), satellite count, and signal quality (fix_type). |
| 34 | `vehicle_land_detected` | 6 | Logic flags for ground contact and landing status. |
| 35 | `vehicle_local_position` | 42 | Position in NED frame (x, y, z) and local velocity. |
| 36 | `vehicle_local_position_setpoint` | 17 | Targeted local coordinates for position holding. |
| 37 | `vehicle_magnetometer` | 3 | Unified magnetic field data for the estimator. |
| 38 | `vehicle_rates_setpoint` | 7 | Desired roll, pitch, and yaw rates. |
| 39 | `vehicle_status` | 31 | Navigation state, arming state, and failsafe status. |
| 40 | `vehicle_status_flags` | 31 | Detailed boolean flags for hardware/software health. |
| 41 | `vehicle_vision_attitude` | 12 | Attitude data provided by external Vision systems. |
| 42 | `vehicle_vision_position` | 42 | Position data provided by external Vision systems. |
| 43 | `vehicle_visual_odometry` | 60 | Raw visual odometry pose and covariance matrices. |
| 44 | `vehicle_visual_odometry_aligned` | 60 | Transformation-aligned visual odometry data. |
