import random


SOURCE_ASSET = "FILE-01"

EXTERNAL_IPS = [
    "198.51.100.24",
    "203.0.113.15",
    "192.0.2.44"
]


def apply_exfiltration_profile(
    event,
    source_asset
):
    event.source_ip = source_asset["ip"]

    event.destination_ip = random.choice(
        EXTERNAL_IPS
    )

    event.source_port = random.randint(
        49152,
        65535
    )

    event.destination_port = random.choice([
        443,
        8443
    ])

    event.protocol = "TCP"

    event.duration = round(
        random.uniform(10.0, 60.0),
        3
    )

    event.bytes_out = random.randint(
        2_000_000,
        25_000_000
    )

    event.bytes_in = random.randint(
        1000,
        100000
    )

    event.packets_out = random.randint(
        1000,
        10000
    )

    event.packets_in = random.randint(
        20,
        300
    )

    event.tcp_flags = "PA"

    event.dns_query = None
    event.dns_entropy = None

    event.http_method = None
    event.http_status = None

    return event