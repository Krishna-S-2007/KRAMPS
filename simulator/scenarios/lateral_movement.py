import random


SOURCE_ASSET = "HOST-07"

LATERAL_TARGETS = [
    "FILE-01",
    "C2-01",
    "COMM-01"
]


def apply_lateral_movement_profile(
    event,
    source_asset,
    target_asset
):
    event.source_ip = source_asset["ip"]
    event.destination_ip = target_asset["ip"]

    event.destination_port = random.choice([
        445,
        3389,
        22
    ])

    event.protocol = "TCP"

    event.duration = round(
        random.uniform(0.5, 8.0),
        3
    )

    event.packets_out = random.randint(15, 120)
    event.packets_in = random.randint(10, 90)

    event.bytes_out = random.randint(
        5000,
        150000
    )

    event.bytes_in = random.randint(
        3000,
        100000
    )

    event.tcp_flags = "PA"

    event.dns_query = None
    event.dns_entropy = None

    event.http_method = None
    event.http_status = None

    return event