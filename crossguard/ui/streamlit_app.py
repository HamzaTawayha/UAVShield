from __future__ import annotations

from dataclasses import asdict
from typing import Any

import streamlit as st

from crossguard.defense.harness import CrossGuardHarness
from crossguard.defense.invariants import InvariantConfig
from crossguard.defense.state import (
    BatteryState,
    Command,
    DroneState,
    GeoPoint,
    PerceptionObject,
    Velocity,
)
from crossguard.utils.geo import horizontal_distance_m


SCENARIOS: dict[str, dict[str, Any]] = {
    "Battery Mirage": {
        "summary": "State of charge is poisoned from full battery to near empty.",
        "prev": {
            "timestamp": 0.0,
            "lat": 42.3314,
            "lon": -83.0458,
            "alt_m": 20.0,
            "battery_pct": 100.0,
            "current_a": 8.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        "curr": {
            "timestamp": 240.0,
            "lat": 42.3314,
            "lon": -83.0458,
            "alt_m": 20.0,
            "battery_pct": 2.0,
            "current_a": 8.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        "extras": {},
    },
    "GPS Jump": {
        "summary": "Telemetry says the drone moved from Michigan to California in 20 minutes.",
        "prev": {
            "timestamp": 0.0,
            "lat": 42.3314,
            "lon": -83.0458,
            "alt_m": 20.0,
            "battery_pct": 100.0,
            "current_a": 8.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        "curr": {
            "timestamp": 1200.0,
            "lat": 34.0522,
            "lon": -118.2437,
            "alt_m": 20.0,
            "battery_pct": 99.0,
            "current_a": 8.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        "extras": {},
    },
    "Phantom Obstacle": {
        "summary": "Perception claims a close obstacle, but depth says free space.",
        "prev": {
            "timestamp": 0.0,
            "lat": 42.3314,
            "lon": -83.0458,
            "alt_m": 20.0,
            "battery_pct": 99.0,
            "current_a": 6.0,
            "vx": 1.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        "curr": {
            "timestamp": 5.0,
            "lat": 42.3314,
            "lon": -83.04574,
            "alt_m": 20.0,
            "battery_pct": 98.9,
            "current_a": 6.0,
            "vx": 1.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        "extras": {
            "perception_enabled": True,
            "claimed_range_m": 3.0,
            "depth_range_m": 15.0,
        },
    },
    "False Waypoint Reached": {
        "summary": "Mission state says a waypoint was reached while the drone is still far away.",
        "prev": {
            "timestamp": 0.0,
            "lat": 0.0,
            "lon": 0.0,
            "alt_m": 5.0,
            "battery_pct": 90.0,
            "current_a": 5.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        "curr": {
            "timestamp": 10.0,
            "lat": 0.0,
            "lon": 0.0,
            "alt_m": 5.0,
            "battery_pct": 89.8,
            "current_a": 5.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        "extras": {
            "waypoint_enabled": True,
            "waypoint_lat": 0.0,
            "waypoint_lon": 0.0001,
            "waypoint_alt_m": 5.0,
            "reported_waypoint_reached": True,
        },
    },
    "Normal State": {
        "summary": "A boring clean transition that should pass.",
        "prev": {
            "timestamp": 0.0,
            "lat": 42.3314,
            "lon": -83.0458,
            "alt_m": 20.0,
            "battery_pct": 100.0,
            "current_a": 6.0,
            "vx": 1.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        "curr": {
            "timestamp": 10.0,
            "lat": 42.3314,
            "lon": -83.04568,
            "alt_m": 20.0,
            "battery_pct": 99.5,
            "current_a": 6.0,
            "vx": 1.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        "extras": {},
    },
}

FIELD_HELP = {
    "timestamp": "Seconds since the start of the experiment or mission. CrossGuard uses the difference between timestamps to compute rates.",
    "battery": "The battery state-of-charge percentage reported to the planner. Poisoning this can cause false aborts or unsafe confidence.",
    "current": "Electrical current draw in amps. Used to sanity-check whether battery depletion is plausible.",
    "lat": "Latitude of the drone's reported position. A poisoned value can make the planner believe the drone teleported.",
    "lon": "Longitude of the drone's reported position. CrossGuard compares position changes against elapsed time.",
    "alt": "Altitude in meters. CrossGuard checks climb/descent rate and command plausibility.",
    "vx": "Reported velocity along the local x/north axis in meters per second.",
    "vy": "Reported velocity along the local y/east axis in meters per second.",
    "vz": "Reported vertical velocity in meters per second.",
    "claimed_range": "Distance to an object according to the perception layer. This is what an attacker might inject.",
    "depth_range": "Independent depth camera reading at the same image location. CrossGuard compares this with the claimed range.",
    "waypoint": "The mission target that the drone claims to have reached.",
    "command": "A planner command, such as goto(target), that CrossGuard checks for physical plausibility.",
    "eta": "Required time-to-arrive. If the target is too far for this ETA, the command is suspicious.",
}

CHECK_EXPLANATIONS = {
    "battery.impossible_drop": "The reported battery percentage fell faster than the configured safe discharge rate.",
    "battery.current_drop_mismatch": "Battery dropped even though reported current draw is too low to explain that drop.",
    "gps.impossible_jump": "The position change would require the drone to travel faster than its physical speed limit.",
    "telemetry.velocity_position_mismatch": "The reported velocity does not match the movement implied by GPS/position changes.",
    "perception.depth_mismatch": "Perception and depth disagree about how far away the same object is.",
    "mission.false_waypoint_reached": "The state says the waypoint was reached, but the drone is still outside the allowed radius.",
    "command.required_ground_speed": "The command requires too much horizontal speed to be physically possible.",
    "command.required_vertical_speed": "The command requires too much climb or descent speed.",
    "command.distance": "The command target is farther away than the mission should allow in one step.",
    "command.no_effect": "Enough time passed after a command, but the drone did not make progress toward the target.",
    "sensor.stale": "One or more sensor readings are too old to trust.",
    "sensor.future_timestamp": "A sensor timestamp is in the future, which can indicate replay or clock poisoning.",
    "packet.replay": "The same packet identifier appeared twice.",
    "altitude.ground_range_mismatch": "Altitude and rangefinder/ground-distance readings disagree.",
    "imu.velocity_mismatch": "IMU acceleration does not match the change in reported velocity.",
    "compass.heading_velocity_mismatch": "The drone's heading does not match its direction of travel.",
    "peer.false_waypoint_reached": "A peer drone claims mission progress that its own position does not support.",
    "ml.state_anomaly": "The learned rolling state model says this full UAV state sequence does not look like real flight behavior.",
}


def main() -> None:
    st.set_page_config(
        page_title="CrossGuard Sanity Harness",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="eyebrow">CrossGuard Test Bench</div>
            <h1>Runtime sanity checks for poisoned UAV state</h1>
            <p>Compare a trusted previous state with injected planner-visible state, then run the real rule-based harness.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scenario_name = st.sidebar.selectbox("Scenario", list(SCENARIOS.keys()), index=0)
    scenario = SCENARIOS[scenario_name]
    st.sidebar.caption(scenario["summary"])
    sidebar_explainer()

    alert_threshold = st.sidebar.slider(
        "Alert threshold",
        min_value=1,
        max_value=10,
        value=3,
        help="Total severity needed before CrossGuard escalates from suspicious to alert.",
    )
    cfg = config_controls()

    st.markdown('<div class="section-title">State Inputs</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="explain-box">
          <strong>How to use this:</strong> the left side is what the drone believed before the attack.
          The right side is the injected or current value that the planner is about to trust.
          CrossGuard compares the transition and looks for physical or mission-logic contradictions.
        </div>
        """,
        unsafe_allow_html=True,
    )
    prev_col, curr_col = st.columns(2, gap="large")
    with prev_col:
        st.markdown('<div class="panel-title">Previous known-good state</div>', unsafe_allow_html=True)
        st.caption("Baseline snapshot from before the suspected poisoning event.")
        prev_values = state_controls(f"{scenario_name}-prev", scenario["prev"])
    with curr_col:
        st.markdown('<div class="panel-title">Injected/current state</div>', unsafe_allow_html=True)
        st.caption("The planner-visible state after the suspected injected or corrupted value.")
        curr_values = state_controls(f"{scenario_name}-curr", scenario["curr"])

    extras = extras_controls(scenario_name, scenario.get("extras", {}))
    previous_state, current_state = build_states(prev_values, curr_values, extras)
    decision = run_harness(previous_state, current_state, cfg, alert_threshold)

    st.markdown('<div class="section-title">What CrossGuard Computed</div>', unsafe_allow_html=True)
    render_transition_summary(previous_state, current_state)

    st.markdown('<div class="section-title">Verdict</div>', unsafe_allow_html=True)
    render_verdict(decision, alert_threshold)

    detail_col, raw_col = st.columns([1.25, 1], gap="large")
    with detail_col:
        render_violations(decision)
    with raw_col:
        render_state_preview(previous_state, current_state)


def sidebar_explainer() -> None:
    with st.sidebar.expander("How the demo works", expanded=True):
        st.markdown(
            """
            1. Pick a scenario.
            2. Edit the previous and injected state.
            3. CrossGuard computes rates and consistency checks.
            4. The verdict shows whether the state should be trusted.
            """
        )

    with st.sidebar.expander("Field guide", expanded=False):
        st.markdown(
            """
            - **Timestamp:** lets CrossGuard calculate rates.
            - **Battery:** reported charge percentage.
            - **Current draw:** helps judge whether battery drop is plausible.
            - **Latitude/Longitude:** reported GPS position.
            - **Altitude:** reported height in meters.
            - **Velocity:** reported movement along x/y/z axes.
            - **Claimed range:** perception's object-distance claim.
            - **Depth reading:** independent sensor distance at that pixel.
            - **Waypoint:** target the drone claims it reached.
            - **Command ETA:** how quickly the planner expects the drone to arrive.
            """
        )


def config_controls() -> InvariantConfig:
    with st.sidebar.expander("Rule thresholds", expanded=False):
        max_ground_speed = st.number_input(
            "Max ground speed (m/s)",
            min_value=1.0,
            value=35.0,
            help="If position changes imply a faster speed, CrossGuard flags an impossible GPS jump.",
        )
        max_battery_drop = st.number_input(
            "Max battery drop (%/min)",
            min_value=0.1,
            value=5.0,
            help="A larger state-of-charge drop per minute is treated as battery-state poisoning.",
        )
        depth_tolerance = st.number_input(
            "Perception-depth tolerance (m)",
            min_value=0.1,
            value=2.0,
            help="Maximum allowed difference between claimed object distance and depth-camera distance.",
        )
        waypoint_radius = st.number_input(
            "Waypoint radius (m)",
            min_value=0.1,
            value=3.0,
            help="The drone must be within this radius to honestly claim waypoint reached.",
        )
        velocity_residual = st.number_input(
            "Velocity-position tolerance (m/s)",
            min_value=0.1,
            value=8.0,
            help="Allowed mismatch between reported velocity and position-derived velocity.",
        )
    return InvariantConfig(
        max_ground_speed_mps=max_ground_speed,
        max_battery_drop_pct_per_min=max_battery_drop,
        perception_depth_tolerance_m=depth_tolerance,
        waypoint_radius_m=waypoint_radius,
        velocity_position_tolerance_mps=velocity_residual,
    )


def state_controls(key_prefix: str, defaults: dict[str, float]) -> dict[str, float]:
    top = st.columns([1, 1, 1])
    timestamp = top[0].number_input(
        "Timestamp (s)",
        value=float(defaults["timestamp"]),
        key=f"{key_prefix}-timestamp",
        help=FIELD_HELP["timestamp"],
    )
    battery = top[1].number_input(
        "Battery (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(defaults["battery_pct"]),
        key=f"{key_prefix}-battery",
        help=FIELD_HELP["battery"],
    )
    current = top[2].number_input(
        "Current draw (A)",
        value=float(defaults["current_a"]),
        key=f"{key_prefix}-current",
        help=FIELD_HELP["current"],
    )

    loc = st.columns([1, 1, 1])
    lat = loc[0].number_input(
        "Latitude",
        value=float(defaults["lat"]),
        format="%.7f",
        key=f"{key_prefix}-lat",
        help=FIELD_HELP["lat"],
    )
    lon = loc[1].number_input(
        "Longitude",
        value=float(defaults["lon"]),
        format="%.7f",
        key=f"{key_prefix}-lon",
        help=FIELD_HELP["lon"],
    )
    alt = loc[2].number_input(
        "Altitude (m)",
        value=float(defaults["alt_m"]),
        key=f"{key_prefix}-alt",
        help=FIELD_HELP["alt"],
    )

    with st.expander("Velocity", expanded=False):
        st.caption("Velocity is optional telemetry that helps catch drift or mismatch attacks.")
        vel_cols = st.columns(3)
        vx = vel_cols[0].number_input("vx (m/s)", value=float(defaults["vx"]), key=f"{key_prefix}-vx", help=FIELD_HELP["vx"])
        vy = vel_cols[1].number_input("vy (m/s)", value=float(defaults["vy"]), key=f"{key_prefix}-vy", help=FIELD_HELP["vy"])
        vz = vel_cols[2].number_input("vz (m/s)", value=float(defaults["vz"]), key=f"{key_prefix}-vz", help=FIELD_HELP["vz"])

    return {
        "timestamp": timestamp,
        "battery_pct": battery,
        "current_a": current,
        "lat": lat,
        "lon": lon,
        "alt_m": alt,
        "vx": vx,
        "vy": vy,
        "vz": vz,
    }


def extras_controls(scenario_name: str, defaults: dict[str, Any]) -> dict[str, Any]:
    st.markdown('<div class="section-title">Optional Attack Fields</div>', unsafe_allow_html=True)
    p_col, w_col, c_col = st.columns(3, gap="large")

    with p_col:
        st.markdown('<div class="panel-title small">Perception vs depth</div>', unsafe_allow_html=True)
        st.caption("Use this to simulate a fake obstacle or object-distance poisoning.")
        perception_enabled = st.checkbox(
            "Inject object detection",
            value=bool(defaults.get("perception_enabled", False)),
            key=f"{scenario_name}-perception-enabled",
            help="Adds a perception object to the injected state.",
        )
        claimed_range = st.number_input(
            "Claimed range (m)",
            min_value=0.0,
            value=float(defaults.get("claimed_range_m", 3.0)),
            key=f"{scenario_name}-claimed-range",
            disabled=not perception_enabled,
            help=FIELD_HELP["claimed_range"],
        )
        depth_range = st.number_input(
            "Depth reading (m)",
            min_value=0.0,
            value=float(defaults.get("depth_range_m", 15.0)),
            key=f"{scenario_name}-depth-range",
            disabled=not perception_enabled,
            help=FIELD_HELP["depth_range"],
        )

    with w_col:
        st.markdown('<div class="panel-title small">Waypoint claim</div>', unsafe_allow_html=True)
        st.caption("Use this to test a false mission-progress claim.")
        waypoint_enabled = st.checkbox(
            "Include waypoint status",
            value=bool(defaults.get("waypoint_enabled", False)),
            key=f"{scenario_name}-waypoint-enabled",
            help="Adds a current waypoint and reached/not-reached claim.",
        )
        reached = st.checkbox(
            "Reported reached",
            value=bool(defaults.get("reported_waypoint_reached", False)),
            key=f"{scenario_name}-waypoint-reached",
            disabled=not waypoint_enabled,
            help="If checked, CrossGuard verifies that the drone is actually near the waypoint.",
        )
        wp_lat = st.number_input("Waypoint lat", value=float(defaults.get("waypoint_lat", 0.0)), format="%.7f", key=f"{scenario_name}-wp-lat", disabled=not waypoint_enabled, help=FIELD_HELP["waypoint"])
        wp_lon = st.number_input("Waypoint lon", value=float(defaults.get("waypoint_lon", 0.0)), format="%.7f", key=f"{scenario_name}-wp-lon", disabled=not waypoint_enabled, help=FIELD_HELP["waypoint"])
        wp_alt = st.number_input("Waypoint alt (m)", value=float(defaults.get("waypoint_alt_m", 5.0)), key=f"{scenario_name}-wp-alt", disabled=not waypoint_enabled, help=FIELD_HELP["waypoint"])

    with c_col:
        st.markdown('<div class="panel-title small">Command plausibility</div>', unsafe_allow_html=True)
        st.caption("Use this to test whether a planner command asks for impossible movement.")
        command_enabled = st.checkbox(
            "Include goto command",
            value=bool(defaults.get("command_enabled", False)),
            key=f"{scenario_name}-command-enabled",
            help="Adds a goto command target to the injected state.",
        )
        cmd_lat = st.number_input("Target lat", value=float(defaults.get("command_lat", 0.0)), format="%.7f", key=f"{scenario_name}-cmd-lat", disabled=not command_enabled, help=FIELD_HELP["command"])
        cmd_lon = st.number_input("Target lon", value=float(defaults.get("command_lon", 0.001)), format="%.7f", key=f"{scenario_name}-cmd-lon", disabled=not command_enabled, help=FIELD_HELP["command"])
        cmd_alt = st.number_input("Target alt (m)", value=float(defaults.get("command_alt_m", 20.0)), key=f"{scenario_name}-cmd-alt", disabled=not command_enabled, help=FIELD_HELP["command"])
        eta_s = st.number_input("Required ETA (s)", min_value=0.1, value=float(defaults.get("eta_s", 10.0)), key=f"{scenario_name}-eta", disabled=not command_enabled, help=FIELD_HELP["eta"])

    return {
        "perception_enabled": perception_enabled,
        "claimed_range_m": claimed_range,
        "depth_range_m": depth_range,
        "waypoint_enabled": waypoint_enabled,
        "reported_waypoint_reached": reached,
        "waypoint_lat": wp_lat,
        "waypoint_lon": wp_lon,
        "waypoint_alt_m": wp_alt,
        "command_enabled": command_enabled,
        "command_lat": cmd_lat,
        "command_lon": cmd_lon,
        "command_alt_m": cmd_alt,
        "eta_s": eta_s,
    }


def build_states(prev_values: dict[str, float], curr_values: dict[str, float], extras: dict[str, Any]) -> tuple[DroneState, DroneState]:
    previous = values_to_state(prev_values)
    perception_objects: tuple[PerceptionObject, ...] = ()
    if extras["perception_enabled"]:
        perception_objects = (
            PerceptionObject(
                class_id="obstacle",
                bbox_center_x=320.0,
                bbox_center_y=240.0,
                confidence=0.92,
                claimed_range_m=float(extras["claimed_range_m"]),
                depth_range_m=float(extras["depth_range_m"]),
            ),
        )

    waypoint = None
    if extras["waypoint_enabled"]:
        waypoint = GeoPoint(float(extras["waypoint_lat"]), float(extras["waypoint_lon"]), float(extras["waypoint_alt_m"]))

    command = None
    if extras["command_enabled"]:
        command = Command(
            command_type="goto",
            issued_at=float(curr_values["timestamp"]),
            target=GeoPoint(float(extras["command_lat"]), float(extras["command_lon"]), float(extras["command_alt_m"])),
            metadata={"eta_s": float(extras["eta_s"])},
        )

    current = values_to_state(
        curr_values,
        perception_objects=perception_objects,
        current_waypoint=waypoint,
        reported_waypoint_reached=bool(extras["reported_waypoint_reached"]),
        last_command=command,
    )
    return previous, current


def values_to_state(
    values: dict[str, float],
    perception_objects: tuple[PerceptionObject, ...] = (),
    current_waypoint: GeoPoint | None = None,
    reported_waypoint_reached: bool = False,
    last_command: Command | None = None,
) -> DroneState:
    return DroneState(
        timestamp=float(values["timestamp"]),
        position=GeoPoint(float(values["lat"]), float(values["lon"]), float(values["alt_m"])),
        velocity=Velocity(float(values["vx"]), float(values["vy"]), float(values["vz"])),
        battery=BatteryState(float(values["battery_pct"]), current_a=float(values["current_a"])),
        perception_objects=perception_objects,
        current_waypoint=current_waypoint,
        reported_waypoint_reached=reported_waypoint_reached,
        last_command=last_command,
        source="streamlit",
    )


def run_harness(previous_state: DroneState, current_state: DroneState, cfg: InvariantConfig, alert_threshold: int):
    harness = CrossGuardHarness(config=cfg, alert_threshold=alert_threshold)
    harness.observe(previous_state)
    return harness.observe(current_state)


def render_transition_summary(previous_state: DroneState, current_state: DroneState) -> None:
    dt_s = current_state.timestamp - previous_state.timestamp
    ground_distance = horizontal_distance_m(previous_state.position, current_state.position)
    required_speed = ground_distance / dt_s if dt_s > 0 else float("inf")
    altitude_delta = current_state.position.alt_m - previous_state.position.alt_m
    battery_delta = None
    battery_rate = None
    if previous_state.battery and current_state.battery and dt_s > 0:
        battery_delta = current_state.battery.percent - previous_state.battery.percent
        battery_rate = -battery_delta / (dt_s / 60.0)

    cards = st.columns(4)
    cards[0].markdown(
        f'<div class="metric-card"><span>Elapsed Time</span><strong>{dt_s:.1f}s</strong><small>Time between snapshots</small></div>',
        unsafe_allow_html=True,
    )
    cards[1].markdown(
        f'<div class="metric-card"><span>Ground Distance</span><strong>{ground_distance:.1f}m</strong><small>Movement implied by lat/lon</small></div>',
        unsafe_allow_html=True,
    )
    cards[2].markdown(
        f'<div class="metric-card"><span>Required Speed</span><strong>{required_speed:.1f}</strong><small>m/s needed to explain movement</small></div>',
        unsafe_allow_html=True,
    )
    battery_text = "n/a" if battery_rate is None else f"{battery_rate:.1f}"
    battery_small = "No battery inputs" if battery_rate is None else "%/min discharge rate"
    cards[3].markdown(
        f'<div class="metric-card"><span>Battery Drop Rate</span><strong>{battery_text}</strong><small>{battery_small}</small></div>',
        unsafe_allow_html=True,
    )

    st.caption(
        f"Altitude changed by {altitude_delta:.1f} m. "
        "These computed values are what the rule checks compare against the configured thresholds."
    )


def render_verdict(decision, alert_threshold: int) -> None:
    severity_total = sum(v.severity for v in decision.violations)
    if decision.alert:
        label = "ALERT"
        tone = "danger"
        description = "CrossGuard would request hover or block planner trust."
    elif decision.violations:
        label = "SUSPICIOUS"
        tone = "warn"
        description = "A rule fired, but the score did not cross the alert threshold yet."
    else:
        label = "PASS"
        tone = "safe"
        description = "No sanity violation was detected for this transition."

    cards = st.columns(4)
    cards[0].markdown(f'<div class="metric-card {tone}"><span>Verdict</span><strong>{label}</strong><small>{description}</small></div>', unsafe_allow_html=True)
    cards[1].markdown(f'<div class="metric-card"><span>Violations</span><strong>{len(decision.violations)}</strong><small>Rules triggered</small></div>', unsafe_allow_html=True)
    cards[2].markdown(f'<div class="metric-card"><span>Severity</span><strong>{severity_total}</strong><small>Alert threshold: {alert_threshold}</small></div>', unsafe_allow_html=True)
    cards[3].markdown(f'<div class="metric-card"><span>Suspicion</span><strong>{decision.suspicion}</strong><small>Post-decision score</small></div>', unsafe_allow_html=True)


def render_violations(decision) -> None:
    st.markdown('<div class="panel-title">Triggered Rules</div>', unsafe_allow_html=True)
    if not decision.violations:
        st.success("No violations. This state transition is plausible under the current thresholds.")
        return

    rows = [
        {
            "check": violation.check_id,
            "severity": violation.severity,
            "observed": violation.observed,
            "threshold": violation.threshold,
            "message": violation.message,
            "plain English": explain_violation(violation.check_id),
        }
        for violation in decision.violations
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Why these rules fired", expanded=True):
        for violation in decision.violations:
            st.markdown(
                f"""
                <div class="rule-card">
                  <strong>{violation.check_id}</strong>
                  <p>{explain_violation(violation.check_id)}</p>
                  <small>{violation.message}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )


def explain_violation(check_id: str) -> str:
    return CHECK_EXPLANATIONS.get(check_id, "This rule detected a contradiction between the supplied state and the configured safety envelope.")


def render_state_preview(previous_state: DroneState, current_state: DroneState) -> None:
    st.markdown('<div class="panel-title">State Objects</div>', unsafe_allow_html=True)
    st.caption("This is the exact normalized object passed into the algorithm.")
    tab_prev, tab_curr = st.tabs(["Previous", "Injected"])
    with tab_prev:
        st.json(asdict(previous_state))
    with tab_curr:
        st.json(asdict(current_state))


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f8fb;
            color: #172033;
        }
        .hero {
            background: #ffffff;
            border: 1px solid #d8e0ea;
            border-left: 6px solid #0f766e;
            border-radius: 8px;
            padding: 24px 28px;
            margin-bottom: 18px;
            box-shadow: 0 10px 30px rgba(23, 32, 51, 0.06);
        }
        .hero h1 {
            font-size: 34px;
            line-height: 1.15;
            margin: 4px 0 8px 0;
            letter-spacing: 0;
            color: #111827;
        }
        .hero p {
            margin: 0;
            color: #526173;
            font-size: 16px;
        }
        .eyebrow {
            color: #0f766e;
            font-weight: 700;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .section-title {
            font-weight: 750;
            font-size: 18px;
            color: #111827;
            margin: 16px 0 8px;
        }
        .panel-title {
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 8px;
        }
        .panel-title.small {
            font-size: 15px;
        }
        .metric-card {
            min-height: 128px;
            border-radius: 8px;
            padding: 16px;
            border: 1px solid #d8e0ea;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(23, 32, 51, 0.05);
        }
        .metric-card span {
            display: block;
            color: #526173;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .metric-card strong {
            display: block;
            color: #111827;
            font-size: 28px;
            line-height: 1.2;
            margin-top: 6px;
        }
        .metric-card small {
            display: block;
            color: #526173;
            margin-top: 8px;
            line-height: 1.35;
        }
        .metric-card.safe {
            border-left: 6px solid #15803d;
        }
        .metric-card.warn {
            border-left: 6px solid #b45309;
        }
        .metric-card.danger {
            border-left: 6px solid #b91c1c;
        }
        .explain-box {
            background: #ffffff;
            border: 1px solid #d8e0ea;
            border-left: 6px solid #2563eb;
            border-radius: 8px;
            padding: 12px 14px;
            color: #334155;
            margin: 8px 0 14px;
            box-shadow: 0 6px 16px rgba(23, 32, 51, 0.04);
        }
        .rule-card {
            background: #ffffff;
            border: 1px solid #d8e0ea;
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 10px;
        }
        .rule-card strong {
            color: #111827;
            font-size: 15px;
        }
        .rule-card p {
            margin: 6px 0;
            color: #334155;
        }
        .rule-card small {
            color: #64748b;
        }
        div[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #d8e0ea;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d8e0ea;
            border-radius: 8px;
            padding: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
