from datetime import datetime, timezone

from backend.app.telemetry.schemas import TelemetryEvent
from backend.app.telemetry.producer import (
    publish_telemetry,
    close_producer,
)

def generate_normal_network_event() -> TelemetryEvent:

    event = TelemetryEvent(
        timestamp=datetime.now(timezone.utc),

        source_ip="10.10.7.21",
        destination_ip="10.10.2.15",

        source_port=49231,
        destination_port=443,

        protocol="TCP",

        duration=2.41,

        bytes_in=1203,
        bytes_out=48291,

        packets_in=8,
        packets_out=31,

        tcp_flags="PA",

        dns_query=None,
        dns_entropy=None,

        http_method="GET",
        http_status=200
    )

    return event


if __name__ == "__main__":

    event = generate_normal_network_event()

    print("PS14 NETWORK TELEMETRY")
    print("----------------------")
    print(event.model_dump_json(indent=2))

    publish_telemetry(
        event.model_dump(mode="json")
    )

    close_producer()