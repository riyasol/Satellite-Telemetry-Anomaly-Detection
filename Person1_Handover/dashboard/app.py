from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path


st.set_page_config(
    page_title="ESA Mission 1 | B1-B4",
    page_icon="🛰️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_splits(path: str, modified_ns: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


@st.cache_data(show_spinner=False)
def load_table(path: str, modified_ns: int) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_if_present(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return load_table(str(path), path.stat().st_mtime_ns)


def _number(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _show_metric_pair(
    label: str, baseline: pd.Series, candidate: pd.Series, field: str
) -> None:
    before_value = pd.to_numeric(
        pd.Series([baseline[field]]), errors="coerce"
    ).iloc[0]
    after_value = pd.to_numeric(
        pd.Series([candidate[field]]), errors="coerce"
    ).iloc[0]
    if pd.isna(before_value) or pd.isna(after_value):
        st.metric(label, "—", help="Metric is undefined for this scenario.")
        return
    before = float(before_value)
    after = float(after_value)
    st.metric(
        label,
        _number(after),
        delta=_number(after - before),
        help="Delta is B4 minus B1; positive is not always better (for FAR, lower is better).",
    )


def main() -> None:
    st.title("ESA Mission 1 | Satellite Telemetry Anomaly Detection")
    st.caption(
        "B1-B4 results and diagnostics from the saved experiment outputs. "
        "This dashboard displays measured results; it does not tune thresholds "
        "or use test labels to make operational decisions."
    )

    cfg = load_config()
    data_path = (
        resolve_path(cfg["data"]["esa_work_dir"])
        / "mission1"
        / "splits.npz"
    )
    results_dir = resolve_path(cfg["output"]["results_dir"]) / "mission1"
    plots_dir = resolve_path(cfg["output"]["plot_dir"]) / "mission1"
    comparison_path = results_dir / "b1_b4_scenario_comparison.csv"
    ablation_path = results_dir / "ablation_study_results.csv"
    masking_path = results_dir / "masking_ablation_results.csv"
    trace_path = plots_dir / "b4_s3_trace.csv"

    splits = (
        load_splits(str(data_path), data_path.stat().st_mtime_ns)
        if data_path.exists()
        else None
    )
    comparison = _read_if_present(comparison_path)
    ablation = _read_if_present(ablation_path)
    masking = _read_if_present(masking_path)
    trace = _read_if_present(trace_path)

    if comparison is None:
        st.warning(
            f"Benchmark CSV not found: {comparison_path}. "
            "Run the benchmark before opening the arena tab."
        )
    elif (
        "false_alarm_rate" not in comparison.columns
        and {"fp_events", "evaluated_windows"}.issubset(comparison.columns)
    ):
        comparison["false_alarm_rate"] = (
            comparison["fp_events"]
            / comparison["evaluated_windows"].clip(lower=1)
        )
    if splits is None:
        st.warning(
            f"Processed Mission 1 split not found: {data_path}. "
            "The telemetry viewer needs this file."
        )

    telemetry_tab, decision_tab, arena_tab, recovery_tab, safety_tab = st.tabs(
        [
            "Telemetry & Anomalies",
            "Decision Layer",
            "Benchmark Arena",
            "Residual Recovery",
            "Ablation & Masking",
        ]
    )

    with telemetry_tab:
        st.subheader("Telemetry and annotation overlay")
        if splits is None:
            st.info("Add the processed Mission 1 splits to enable this viewer.")
        else:
            channel_names = [
                str(name) for name in splits.get(
                    "channel_names",
                    np.array([f"channel_{index}" for index in range(
                        splits["values"].shape[1]
                    )]),
                )
            ]
            split_name = st.selectbox(
                "Data split", ["train", "validation", "test"], index=2
            )
            bounds = splits[f"{split_name}_bounds"]
            start_bound, end_bound = int(bounds[0]), int(bounds[1])
            max_points = max(100, min(5000, end_bound - start_bound))
            point_options = [
                value for value in (250, 500, 1000, 2000, 5000)
                if value <= max_points
            ]
            if not point_options:
                point_options = [max_points]
            default_points = min(1000, point_options[-1])
            point_count = st.select_slider(
                "Observations to display",
                options=point_options,
                value=default_points,
            )
            last_start = max(start_bound, end_bound - point_count)
            start_index = st.slider(
                "Window start observation",
                min_value=start_bound,
                max_value=last_start,
                value=start_bound,
                step=max(1, point_count // 10),
            )
            channel = st.selectbox("Telemetry channel", channel_names)
            channel_index = channel_names.index(channel)
            stop_index = min(end_bound, start_index + point_count)
            point_slice = slice(start_index, stop_index)

            timestamps = pd.to_datetime(
                splits["timestamps_ns"][point_slice], unit="ns", utc=True
            )
            telemetry = pd.DataFrame(
                {channel: splits["values"][point_slice, channel_index]},
                index=timestamps,
            )
            st.line_chart(telemetry, height=300)

            masks = pd.DataFrame(
                {
                    "Exact anomaly label": splits["labels"][point_slice].astype(
                        np.uint8
                    ),
                    "Safety exclusion mask": splits["exclusion_mask"][
                        point_slice
                    ].astype(np.uint8),
                },
                index=timestamps,
            )
            st.line_chart(masks, height=180)
            st.caption(
                "The exact label and expanded safety-exclusion mask are shown "
                "separately. The mask is not used as an online ground-truth signal."
            )
            metric_columns = st.columns(3)
            metric_columns[0].metric("Displayed samples", stop_index - start_index)
            metric_columns[1].metric(
                "Exact anomaly samples",
                int(splits["labels"][point_slice].sum()),
            )
            metric_columns[2].metric(
                "Safety-masked samples",
                int(splits["exclusion_mask"][point_slice].sum()),
            )

    with decision_tab:
        st.subheader("B4 decision-layer trace")
        if trace is None:
            st.warning(
                f"Decision trace not found: {trace_path}. "
                "Generate it with `python -m src.experiments.plot_drift_timeline`."
            )
        else:
            trace = trace.sort_values("window_step").reset_index(drop=True)
            st.caption(
                "Scenario 3 adaptation stream. The pulse-overlap indicator is "
                "offline audit metadata, not an input to B4 decisions."
            )
            st.line_chart(
                trace.set_index("window_step")[
                    ["b1_raw_mse", "b4_raw_mse", "b1_threshold", "b4_threshold"]
                ],
                height=300,
            )
            final_row = trace.iloc[-1]
            metrics = st.columns(4)
            metrics[0].metric("Latest severity", _number(final_row["severity"]))
            metrics[1].metric(
                "Persistence count", int(final_row["persistence_count"])
            )
            metrics[2].metric("Latest action", str(final_row["action"]))
            metrics[3].metric(
                "Confirmed drift alarms",
                int(trace["confirmed_drift_alarm"].sum()),
            )
            st.progress(
                min(1.0, max(0.0, float(final_row["severity"]))),
                text="Latest severity",
            )

            severity_frame = trace.set_index("window_step")[
                ["severity", "persistence_count", "relative_degradation"]
            ]
            st.line_chart(severity_frame, height=240)
            st.markdown("**Distribution measures**")
            distribution_columns = st.columns(3)
            for column, metric in zip(
                distribution_columns, ("ks", "wasserstein", "js")
            ):
                column.line_chart(
                    trace.set_index("window_step")[[metric]], height=180
                )

            alarm_rows = trace[
                (trace["raw_adwin_alarm"] > 0)
                | (trace["confirmed_drift_alarm"] > 0)
            ]
            st.markdown("**ADWIN alarms and cut points**")
            st.dataframe(
                alarm_rows[
                    [
                        "window_step",
                        "raw_adwin_alarm",
                        "confirmed_drift_alarm",
                        "adwin_cut_stream_index",
                        "mask_dominance_fraction",
                        "suppression_reason",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("**Update actions**")
            action_rows = trace[
                (trace["action"] != "NO_UPDATE")
                | (trace["accepted_update"] > 0)
            ]
            st.dataframe(
                action_rows[
                    [
                        "window_step",
                        "severity",
                        "persistence_count",
                        "persistence_passed",
                        "relative_degradation",
                        "degradation_passed",
                        "action",
                        "accepted_update",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with arena_tab:
        st.subheader("B1-B4 held-out benchmark")
        if comparison is None or comparison.empty:
            st.info("No comparison results are available yet.")
        else:
            scenario_names = comparison["scenario_name"].drop_duplicates().tolist()
            selected_scenario = st.selectbox(
                "Scenario", scenario_names, key="arena_scenario"
            )
            rows = comparison[
                comparison["scenario_name"] == selected_scenario
            ].set_index("baseline")
            if "B1" in rows.index and "B4" in rows.index:
                before, after = rows.loc["B1"], rows.loc["B4"]
                columns = st.columns(4)
                for column, label, field in zip(
                    columns,
                    ("Event F0.5", "Precision", "Recall", "Event FAR"),
                    ("f0.5_event", "precision", "recall", "false_alarm_rate"),
                ):
                    with column:
                        _show_metric_pair(label, before, after, field)
                st.caption(
                    "Metric deltas are B4 minus B1. Lower event FAR is preferable. "
                    "A tie is shown as a tie."
                )

            chart_metrics = [
                metric for metric in
                ("f0.5_event", "precision", "recall", "false_alarm_rate")
                if metric in comparison.columns
            ]
            st.bar_chart(
                comparison[comparison["scenario_name"] == selected_scenario]
                .set_index("baseline")[chart_metrics],
                height=320,
            )
            st.dataframe(
                comparison[comparison["scenario_name"] == selected_scenario],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "These are controlled synthetic drift scenarios and a single "
                "configured seed; they are not a multi-seed confidence interval."
            )

    with recovery_tab:
        st.subheader("Residual recovery")
        if trace is None:
            st.info(
                "Generate the B1/B4 adaptation trace to view residual recovery."
            )
        else:
            logarithmic = st.checkbox("Display log10 residuals", value=True)
            recovery = trace.set_index("window_step")[
                ["b1_raw_mse", "b4_raw_mse", "b1_threshold", "b4_threshold"]
            ].copy()
            if logarithmic:
                recovery = np.log10(recovery.clip(lower=1e-10))
            st.line_chart(recovery, height=420)
            accepted = trace[trace["accepted_update"] > 0]
            st.metric("Accepted B4 updates in trace", len(accepted))
            if not accepted.empty:
                st.dataframe(
                    accepted[
                        [
                            "window_step",
                            "b4_raw_mse",
                            "b4_threshold",
                            "action",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    with safety_tab:
        st.subheader("Ablation and masking safety")
        if ablation is None or ablation.empty:
            st.warning("Ablation results are missing; run the full experiment first.")
        else:
            if "variant" in ablation.columns and "f0.5_event" in ablation.columns:
                st.bar_chart(
                    ablation.set_index("variant")[["f0.5_event"]],
                    height=280,
                )
            metric_columns = [
                field for field in (
                    "variant",
                    "f0.5_event",
                    "total_updates",
                    "candidates_accepted",
                    "candidates_rejected",
                    "candidate_training_pulse_contaminated_exposures",
                    "accepted_replay_pulse_contaminated_additions",
                )
                if field in ablation.columns
            ]
            st.dataframe(
                ablation[metric_columns],
                use_container_width=True,
                hide_index=True,
            )
            audit_fields = {
                "candidate_training_pulse_contaminated_exposures",
                "accepted_replay_pulse_contaminated_additions",
            }
            if not audit_fields.issubset(ablation.columns):
                st.info(
                    "This ablation CSV predates the offline pulse-overlap audit. "
                    "Rerun the updated experiment to measure candidate/replay "
                    "window overlap. Ground-truth pulse labels are used only for "
                    "this post-run audit."
                )
        if masking is not None and not masking.empty:
            st.markdown("**Masking-only comparison**")
            st.dataframe(masking, use_container_width=True, hide_index=True)


main()
