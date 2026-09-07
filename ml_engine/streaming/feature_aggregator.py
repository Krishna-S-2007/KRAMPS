"""
PS14 Feature Aggregator & CIC-IDS2017 Feature Mapper
===================================================
Aggregates behavioral and temporal statistics across a rolling FlowWindow
and maps them to the exact 78 feature contract required by the trained
PS14_AI XGBoost and Isolation Forest models.

Design principles:
- Grounded in real window statistics: IAT, rate variance, packet distributions,
  flag accumulations, burst dynamics, and flow aggregations.
- Deterministic, NaN-safe numerical calculations.
- Strict preservation of the 78 ML_FEATURE_COLUMNS schema and ordering.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from ml_engine.streaming.flow_state import FlowWindow, NormalizedEvent
except ImportError:
    try:
        from .flow_state import FlowWindow, NormalizedEvent
    except ImportError:
        from ml.flow_state import FlowWindow, NormalizedEvent


logger = logging.getLogger(__name__)

# The exact 78 features expected by the trained PS14_AI model
ML_FEATURE_COLUMNS: List[str] = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

_FLAG_CHARS: Dict[str, str] = {
    "F": "FIN Flag Count",
    "S": "SYN Flag Count",
    "R": "RST Flag Count",
    "P": "PSH Flag Count",
    "A": "ACK Flag Count",
    "U": "URG Flag Count",
    "C": "CWE Flag Count",
    "E": "ECE Flag Count",
}

_TCP_HEADER_BYTES: int = 20
_MIN_SEG_SIZE: int = 20


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den == 0.0 or math.isnan(den) or math.isinf(den):
        return default
    res = num / den
    return res if math.isfinite(res) else default


def _calc_stats(values: List[float]) -> Tuple[float, float, float, float, float]:
    """Return (min, max, mean, std, variance) for a list of values."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if len(values) == 1:
        v = float(values[0])
        return v, v, v, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    v_mean = float(np.mean(arr))
    v_std = float(np.std(arr))
    v_var = float(np.var(arr))
    return v_min, v_max, v_mean, v_std, v_var


class FeatureAggregator:
    """
    Computes statistical feature aggregates from a FlowWindow and maps them
    to the 78 CIC-IDS2017 feature representation.
    """

    def aggregate(
        self,
        current_event: NormalizedEvent,
        window: FlowWindow,
        host_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Extract flow and behavioral metrics from current event and its window history.
        """
        events: List[NormalizedEvent] = list(window.events) if window and window.events else [current_event]

        # ── 1. Aggregated Packet & Byte Counts ──────────────────────────────
        total_fwd_pkts = sum(e.packets_out for e in events)
        total_bwd_pkts = sum(e.packets_in for e in events)
        total_fwd_bytes = sum(e.bytes_out for e in events)
        total_bwd_bytes = sum(e.bytes_in for e in events)

        # ── 2. Flow Duration (microseconds) ─────────────────────────────────
        if len(events) > 1:
            total_duration_s = max(events[-1].timestamp - events[0].timestamp, sum(e.duration for e in events) / len(events))
        else:
            total_duration_s = max(current_event.duration, 0.001)

        flow_duration_us = total_duration_s * 1_000_000.0

        # ── 3. Packet Length Distributions ──────────────────────────────────
        fwd_pkt_lengths: List[float] = []
        bwd_pkt_lengths: List[float] = []
        all_pkt_lengths: List[float] = []

        for e in events:
            if e.packets_out > 0:
                mean_fwd = _safe_div(e.bytes_out, e.packets_out)
                fwd_pkt_lengths.append(mean_fwd)
                all_pkt_lengths.append(mean_fwd)
                # Model realistic packet distribution across control (54B TCP/28B UDP) and data frames
                if e.packets_out > 1 and e.bytes_out > e.packets_out * 60:
                    min_ctrl = 54.0 if e.protocol == "TCP" else 28.0
                    payload_pkts = max(e.packets_out - 2, 1)
                    payload_mean = max((e.bytes_out - 2 * min_ctrl) / payload_pkts, min_ctrl)
                    fwd_pkt_lengths.extend([min_ctrl, payload_mean])
                    all_pkt_lengths.extend([min_ctrl, payload_mean])

            if e.packets_in > 0:
                mean_bwd = _safe_div(e.bytes_in, e.packets_in)
                bwd_pkt_lengths.append(mean_bwd)
                all_pkt_lengths.append(mean_bwd)
                if e.packets_in > 1 and e.bytes_in > e.packets_in * 60:
                    min_ctrl = 54.0 if e.protocol == "TCP" else 28.0
                    payload_pkts = max(e.packets_in - 2, 1)
                    payload_mean = max((e.bytes_in - 2 * min_ctrl) / payload_pkts, min_ctrl)
                    bwd_pkt_lengths.extend([min_ctrl, payload_mean])
                    all_pkt_lengths.extend([min_ctrl, payload_mean])

        f_min, f_max, f_mean, f_std, _ = _calc_stats(fwd_pkt_lengths)
        b_min, b_max, b_mean, b_std, _ = _calc_stats(bwd_pkt_lengths)
        all_min, all_max, all_mean, all_std, all_var = _calc_stats(all_pkt_lengths)

        # ── 4. Flow & Packet Rates ──────────────────────────────────────────
        all_bytes = total_fwd_bytes + total_bwd_bytes
        all_pkts = total_fwd_pkts + total_bwd_pkts

        flow_bytes_s = _safe_div(all_bytes, total_duration_s)
        flow_pkts_s = _safe_div(all_pkts, total_duration_s)
        fwd_pkts_s = _safe_div(total_fwd_pkts, total_duration_s)
        bwd_pkts_s = _safe_div(total_bwd_pkts, total_duration_s)

        # ── 5. Inter-Arrival Times (IAT) from Window History ─────────────────
        flow_iats: List[float] = []
        fwd_iats: List[float] = []
        bwd_iats: List[float] = []

        fwd_events = [e for e in events if e.packets_out > 0]
        bwd_events = [e for e in events if e.packets_in > 0]

        for i in range(1, len(events)):
            delta_us = max((events[i].timestamp - events[i - 1].timestamp) * 1_000_000.0, 0.0)
            flow_iats.append(delta_us)

        for i in range(1, len(fwd_events)):
            delta_us = max((fwd_events[i].timestamp - fwd_events[i - 1].timestamp) * 1_000_000.0, 0.0)
            fwd_iats.append(delta_us)

        for i in range(1, len(bwd_events)):
            delta_us = max((bwd_events[i].timestamp - bwd_events[i - 1].timestamp) * 1_000_000.0, 0.0)
            bwd_iats.append(delta_us)

        # If only 1 event, estimate within-flow packet IAT
        if not flow_iats:
            est_flow_iat = _safe_div(flow_duration_us, max(all_pkts - 1, 1))
            flow_iat_min, flow_iat_max, flow_iat_mean, flow_iat_std = 0.0, flow_duration_us, est_flow_iat, 0.0
        else:
            flow_iat_min, flow_iat_max, flow_iat_mean, flow_iat_std, _ = _calc_stats(flow_iats)

        if not fwd_iats:
            est_fwd_iat = _safe_div(flow_duration_us, max(total_fwd_pkts - 1, 1))
            fwd_iat_tot, fwd_iat_min, fwd_iat_max, fwd_iat_mean, fwd_iat_std = flow_duration_us, 0.0, flow_duration_us, est_fwd_iat, 0.0
        else:
            fwd_iat_min, fwd_iat_max, fwd_iat_mean, fwd_iat_std, _ = _calc_stats(fwd_iats)
            fwd_iat_tot = sum(fwd_iats)

        if not bwd_iats:
            est_bwd_iat = _safe_div(flow_duration_us, max(total_bwd_pkts - 1, 1))
            bwd_iat_tot, bwd_iat_min, bwd_iat_max, bwd_iat_mean, bwd_iat_std = flow_duration_us, 0.0, flow_duration_us, est_bwd_iat, 0.0
        else:
            bwd_iat_min, bwd_iat_max, bwd_iat_mean, bwd_iat_std, _ = _calc_stats(bwd_iats)
            bwd_iat_tot = sum(bwd_iats)

        # ── 6. TCP Flag Accumulations Across Window ─────────────────────────
        flag_counts = {name: 0 for name in _FLAG_CHARS.values()}
        fwd_psh = 0
        fwd_urg = 0

        for e in events:
            flags = (e.tcp_flags or "").upper()
            for ch, name in _FLAG_CHARS.items():
                if ch in flags:
                    flag_counts[name] += 1
            if "P" in flags:
                fwd_psh += 1
            if "U" in flags:
                fwd_urg += 1

        # ── 7. Header Lengths ────────────────────────────────────────────────
        is_tcp = current_event.protocol == "TCP"
        fwd_header_len = total_fwd_pkts * (_TCP_HEADER_BYTES if is_tcp else 8)
        bwd_header_len = total_bwd_pkts * (_TCP_HEADER_BYTES if is_tcp else 8)

        # ── 8. Ratios & Segment Sizes ────────────────────────────────────────
        down_up_ratio = _safe_div(total_bwd_pkts, total_fwd_pkts)
        avg_pkt_size = _safe_div(all_bytes, all_pkts)
        avg_fwd_seg = f_mean
        avg_bwd_seg = b_mean

        # ── 9. Subflows ──────────────────────────────────────────────────────
        subflow_fwd_pkts = total_fwd_pkts
        subflow_fwd_bytes = total_fwd_bytes
        subflow_bwd_pkts = total_bwd_pkts
        subflow_bwd_bytes = total_bwd_bytes

        # ── 10. Active & Idle Dynamics ───────────────────────────────────────
        # An idle period is detected when the gap between consecutive events > 1.0s
        active_periods: List[float] = []
        idle_periods: List[float] = []

        active_accum = 0.0
        for i in range(len(events)):
            active_accum += events[i].duration * 1_000_000.0
            if i > 0:
                gap_s = events[i].timestamp - events[i - 1].timestamp
                if gap_s > 1.0:
                    idle_periods.append(gap_s * 1_000_000.0)
                    active_periods.append(active_accum)
                    active_accum = 0.0

        if active_accum > 0.0:
            active_periods.append(active_accum)

        act_min, act_max, act_mean, act_std, _ = _calc_stats(active_periods)
        idl_min, idl_max, idl_mean, idl_std, _ = _calc_stats(idle_periods)

        # Initial window sizes from payload approximations
        init_win_fwd = max(current_event.bytes_out, 64)
        init_win_bwd = max(current_event.bytes_in, 64) if current_event.packets_in > 0 else 0

        # ── Map to 78 Feature Vector ─────────────────────────────────────────
        row: Dict[str, float] = {
            "Destination Port": float(current_event.destination_port),
            "Flow Duration": float(flow_duration_us),
            "Total Fwd Packets": float(total_fwd_pkts),
            "Total Backward Packets": float(total_bwd_pkts),
            "Total Length of Fwd Packets": float(total_fwd_bytes),
            "Total Length of Bwd Packets": float(total_bwd_bytes),
            "Fwd Packet Length Max": float(f_max),
            "Fwd Packet Length Min": float(f_min),
            "Fwd Packet Length Mean": float(f_mean),
            "Fwd Packet Length Std": float(f_std),
            "Bwd Packet Length Max": float(b_max),
            "Bwd Packet Length Min": float(b_min),
            "Bwd Packet Length Mean": float(b_mean),
            "Bwd Packet Length Std": float(b_std),
            "Flow Bytes/s": float(flow_bytes_s),
            "Flow Packets/s": float(flow_pkts_s),
            "Flow IAT Mean": float(flow_iat_mean),
            "Flow IAT Std": float(flow_iat_std),
            "Flow IAT Max": float(flow_iat_max),
            "Flow IAT Min": float(flow_iat_min),
            "Fwd IAT Total": float(fwd_iat_tot),
            "Fwd IAT Mean": float(fwd_iat_mean),
            "Fwd IAT Std": float(fwd_iat_std),
            "Fwd IAT Max": float(fwd_iat_max),
            "Fwd IAT Min": float(fwd_iat_min),
            "Bwd IAT Total": float(bwd_iat_tot),
            "Bwd IAT Mean": float(bwd_iat_mean),
            "Bwd IAT Std": float(bwd_iat_std),
            "Bwd IAT Max": float(bwd_iat_max),
            "Bwd IAT Min": float(bwd_iat_min),
            "Fwd PSH Flags": float(min(fwd_psh, 1)),
            "Bwd PSH Flags": 0.0,
            "Fwd URG Flags": float(min(fwd_urg, 1)),
            "Bwd URG Flags": 0.0,
            "Fwd Header Length": float(fwd_header_len),
            "Bwd Header Length": float(bwd_header_len),
            "Fwd Packets/s": float(fwd_pkts_s),
            "Bwd Packets/s": float(bwd_pkts_s),
            "Min Packet Length": float(all_min),
            "Max Packet Length": float(all_max),
            "Packet Length Mean": float(all_mean),
            "Packet Length Std": float(all_std),
            "Packet Length Variance": float(all_var),
            "FIN Flag Count": float(flag_counts["FIN Flag Count"]),
            "SYN Flag Count": float(flag_counts["SYN Flag Count"]),
            "RST Flag Count": float(flag_counts["RST Flag Count"]),
            "PSH Flag Count": float(flag_counts["PSH Flag Count"]),
            "ACK Flag Count": float(flag_counts["ACK Flag Count"]),
            "URG Flag Count": float(flag_counts["URG Flag Count"]),
            "CWE Flag Count": float(flag_counts["CWE Flag Count"]),
            "ECE Flag Count": float(flag_counts["ECE Flag Count"]),
            "Down/Up Ratio": float(down_up_ratio),
            "Average Packet Size": float(avg_pkt_size),
            "Avg Fwd Segment Size": float(avg_fwd_seg),
            "Avg Bwd Segment Size": float(avg_bwd_seg),
            "Fwd Header Length.1": float(fwd_header_len),
            "Fwd Avg Bytes/Bulk": 0.0,
            "Fwd Avg Packets/Bulk": 0.0,
            "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0,
            "Bwd Avg Packets/Bulk": 0.0,
            "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": float(subflow_fwd_pkts),
            "Subflow Fwd Bytes": float(subflow_fwd_bytes),
            "Subflow Bwd Packets": float(subflow_bwd_pkts),
            "Subflow Bwd Bytes": float(subflow_bwd_bytes),
            "Init_Win_bytes_forward": float(init_win_fwd),
            "Init_Win_bytes_backward": float(init_win_bwd),
            "act_data_pkt_fwd": float(total_fwd_pkts),
            "min_seg_size_forward": float(_MIN_SEG_SIZE),
            "Active Mean": float(act_mean),
            "Active Std": float(act_std),
            "Active Max": float(act_max),
            "Active Min": float(act_min),
            "Idle Mean": float(idl_mean),
            "Idle Std": float(idl_std),
            "Idle Max": float(idl_max),
            "Idle Min": float(idl_min),
        }

        # Return ordered by ML_FEATURE_COLUMNS
        return {col: row[col] for col in ML_FEATURE_COLUMNS}
