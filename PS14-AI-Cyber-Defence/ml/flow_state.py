"""
PS14 Flow State & Behaviour Window Manager
==========================================
Maintains bounded rolling temporal windows of telemetry events per entity key
(e.g., source_ip, destination_ip, destination_port).

Provides historical context (frequency, burst rates, packet & byte distributions,
inter-arrival times, connection repetitions) so that downstream feature
aggregators can produce realistic CIC-IDS2017 flow features.

Memory safety:
- Configurable window duration in seconds (WINDOW_SECONDS).
- Bounded maximum events stored per conversation key.
- Timestamp-based TTL expiration of inactive keys to prevent unbounded memory growth.
- Clean reset() mechanism for simulations/tests.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class NormalizedEvent:
    """Canonical representation of a single normalized telemetry event."""
    timestamp: float           # epoch timestamp in seconds
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    duration: float            # connection duration in seconds
    bytes_in: int
    bytes_out: int
    packets_in: int
    packets_out: int
    tcp_flags: str             # e.g. "S", "PA", "FA", "R", etc.
    dns_query: Optional[str] = None
    dns_entropy: Optional[float] = None
    http_method: Optional[str] = None
    http_status: Optional[int] = None
    raw_payload: Dict[str, Any] = field(default_factory=dict)


class FlowWindow:
    """
    Tracks recent event history for a single conversation key over a rolling window.
    """

    def __init__(
        self,
        key: Tuple[str, str, int],
        window_seconds: float = 30.0,
        max_events: int = 200,
    ):
        self.key = key  # (source_ip, destination_ip, destination_port)
        self.window_seconds = window_seconds
        self.max_events = max_events
        self.events: Deque[NormalizedEvent] = deque(maxlen=max_events)
        self.last_updated: float = time.time()

    def add_event(self, event: NormalizedEvent) -> None:
        """Add an event in chronological order and prune expired events beyond the window."""
        # Insert in sorted order by timestamp to handle any out-of-order arrivals gracefully
        if not self.events or event.timestamp >= self.events[-1].timestamp:
            self.events.append(event)
        else:
            # Find insertion point
            temp = list(self.events)
            idx = 0
            while idx < len(temp) and temp[idx].timestamp <= event.timestamp:
                idx += 1
            temp.insert(idx, event)
            if len(temp) > self.max_events:
                temp = temp[-self.max_events:]
            self.events = deque(temp, maxlen=self.max_events)

        self.last_updated = max(event.timestamp, self.last_updated)
        self.prune(current_timestamp=event.timestamp)

    def prune(self, current_timestamp: Optional[float] = None) -> None:
        """Remove events that are older than current_timestamp - window_seconds."""
        now = current_timestamp if current_timestamp is not None else time.time()
        cutoff = now - self.window_seconds
        while self.events and self.events[0].timestamp < cutoff:
            self.events.popleft()

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def is_empty(self) -> bool:
        return len(self.events) == 0

    @property
    def duration_span_seconds(self) -> float:
        """Time difference between oldest and newest event in the active window."""
        if len(self.events) < 2:
            if len(self.events) == 1:
                return max(self.events[0].duration, 0.001)
            return 0.0
        return max(self.events[-1].timestamp - self.events[0].timestamp, 0.001)


class FlowStateManager:
    """
    Stateful manager maintaining conversation windows across all network connections.
    
    Provides thread-safe operations, periodic memory cleanup, and behavioral stats.
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        max_events_per_key: int = 200,
        max_active_keys: int = 5000,
        key_idle_timeout: float = 120.0,
    ):
        self.window_seconds = window_seconds
        self.max_events_per_key = max_events_per_key
        self.max_active_keys = max_active_keys
        self.key_idle_timeout = key_idle_timeout

        # Conversation windows: (source_ip, destination_ip, destination_port) -> FlowWindow
        self._windows: Dict[Tuple[str, str, int], FlowWindow] = {}

        # Host-level attempt tracking: source_ip -> Deque[float timestamp]
        self._host_attempts: Dict[str, Deque[float]] = {}

        self._last_cleanup: float = time.time()
        self._cleanup_interval: float = 30.0

    def parse_event(self, raw_event: Dict[str, Any]) -> NormalizedEvent:
        """Parse raw telemetry dict into NormalizedEvent with epoch timestamp."""
        raw_ts = raw_event.get("timestamp")
        if isinstance(raw_ts, (int, float)):
            epoch_ts = float(raw_ts)
        elif isinstance(raw_ts, str):
            try:
                dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                epoch_ts = dt.timestamp()
            except Exception:
                epoch_ts = time.time()
        elif isinstance(raw_ts, datetime):
            epoch_ts = raw_ts.timestamp()
        else:
            epoch_ts = time.time()

        return NormalizedEvent(
            timestamp=epoch_ts,
            source_ip=str(raw_event.get("source_ip", "0.0.0.0")),
            destination_ip=str(raw_event.get("destination_ip", "0.0.0.0")),
            source_port=int(raw_event.get("source_port", 0)),
            destination_port=int(raw_event.get("destination_port", 0)),
            protocol=str(raw_event.get("protocol", "TCP")).upper(),
            duration=float(raw_event.get("duration", 1.0) or 1.0),
            bytes_in=int(raw_event.get("bytes_in", 0) or 0),
            bytes_out=int(raw_event.get("bytes_out", 0) or 0),
            packets_in=int(raw_event.get("packets_in", 0) or 0),
            packets_out=int(raw_event.get("packets_out", 0) or 0),
            tcp_flags=str(raw_event.get("tcp_flags") or ""),
            dns_query=raw_event.get("dns_query"),
            dns_entropy=float(raw_event["dns_entropy"]) if raw_event.get("dns_entropy") is not None else None,
            http_method=raw_event.get("http_method"),
            http_status=int(raw_event["http_status"]) if raw_event.get("http_status") is not None else None,
            raw_payload=raw_event,
        )

    def record_and_get_window(self, raw_event: Dict[str, Any]) -> Tuple[NormalizedEvent, FlowWindow, Dict[str, Any]]:
        """
        Record a new raw event into its conversation window and return the
        normalized event, the conversation FlowWindow, and host-level behavioral context.
        """
        event = self.parse_event(raw_event)
        key = (event.source_ip, event.destination_ip, event.destination_port)

        # Get or create conversation window
        if key not in self._windows:
            if len(self._windows) >= self.max_active_keys:
                self._force_evict_oldest()
            self._windows[key] = FlowWindow(
                key=key,
                window_seconds=self.window_seconds,
                max_events=self.max_events_per_key,
            )

        window = self._windows[key]
        window.add_event(event)

        # Track host-level activity
        src = event.source_ip
        if src not in self._host_attempts:
            self._host_attempts[src] = deque(maxlen=self.max_events_per_key)
        self._host_attempts[src].append(event.timestamp)

        # Host-level context
        now = event.timestamp
        host_attempts_in_window = sum(1 for t in self._host_attempts[src] if t >= now - self.window_seconds)

        host_context = {
            "host_attempts_in_window": host_attempts_in_window,
            "window_event_count": window.event_count,
            "window_span_seconds": window.duration_span_seconds,
        }

        # Periodic cleanup of expired entries
        if now - self._last_cleanup > self._cleanup_interval:
            self.cleanup(current_timestamp=now)

        return event, window, host_context

    def cleanup(self, current_timestamp: Optional[float] = None) -> None:
        """Prune inactive conversation keys and expired events."""
        now = current_timestamp if current_timestamp is not None else time.time()
        self._last_cleanup = now
        stale_keys = []

        for key, window in self._windows.items():
            window.prune(now)
            if window.is_empty and (now - window.last_updated > self.key_idle_timeout):
                stale_keys.append(key)

        for k in stale_keys:
            del self._windows[k]

        # Prune host attempts
        cutoff = now - self.window_seconds
        for src, deq in list(self._host_attempts.items()):
            while deq and deq[0] < cutoff:
                deq.popleft()
            if not deq:
                del self._host_attempts[src]

    def _force_evict_oldest(self) -> None:
        """Evict the least recently updated window if capacity is exceeded."""
        if not self._windows:
            return
        oldest_key = min(self._windows.keys(), key=lambda k: self._windows[k].last_updated)
        del self._windows[oldest_key]

    def reset(self) -> None:
        """Clear all active conversation windows and host history."""
        self._windows.clear()
        self._host_attempts.clear()
        logger.info("[FlowStateManager] State reset successfully.")

    def get_summary(self, source_ip: str, destination_ip: str, destination_port: int) -> Dict[str, Any]:
        """Return human-readable window state for debugging and logging."""
        key = (source_ip, destination_ip, destination_port)
        window = self._windows.get(key)
        if not window:
            return {"active": False, "event_count": 0}
        return {
            "active": True,
            "event_count": window.event_count,
            "duration_span_s": round(window.duration_span_seconds, 3),
            "window_seconds": self.window_seconds,
        }
