from backend.app.telemetry.schemas import TelemetryEvent


def test_network_telemetry_schema():

    event = {
        "timestamp": "2026-08-16T15:10:30Z",

        "source_ip": "10.10.7.21",
        "destination_ip": "10.10.2.15",

        "source_port": 49231,
        "destination_port": 443,

        "protocol": "TCP",

        "duration": 2.41,

        "bytes_in": 1203,
        "bytes_out": 48291,

        "packets_in": 8,
        "packets_out": 31,

        "tcp_flags": "PA",

        "dns_query": None,
        "dns_entropy": None,

        "http_method": "GET",
        "http_status": 200
    }

    telemetry = TelemetryEvent(**event)

    assert telemetry.source_ip == "10.10.7.21"
    assert telemetry.destination_ip == "10.10.2.15"

    assert telemetry.source_port == 49231
    assert telemetry.destination_port == 443

    assert telemetry.protocol == "TCP"

    assert telemetry.bytes_in == 1203
    assert telemetry.bytes_out == 48291

    assert telemetry.packets_in == 8
    assert telemetry.packets_out == 31

    assert telemetry.http_status == 200