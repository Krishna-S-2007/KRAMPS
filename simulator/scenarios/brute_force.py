import random


SOURCE_ASSET = "HOST-07"
TARGET_ASSET = "OPS-01"


def apply_brute_force_profile(
    event,
    source_asset,
    target_asset
):
    """
    Synthetic network-level brute-force behaviour.

    Represents repeated short authentication-related
    connection attempts. No real authentication attack occurs.
    """

    event.source_ip = source_asset["ip"]
    event.destination_ip = target_asset["ip"]

    event.source_port = random.randint(
        49152,
        65535
    )

    # Consistent target service (e.g. SSH on port 22 or RDP on port 3389)
    # targeting a specific authentication service across attempts
    event.destination_port = target_asset.get("auth_port", 22)

    event.protocol = "TCP"

    # Repeated short-lived attempts (typical failed authentication exchanges)
    event.duration = round(
        random.uniform(0.05, 0.4),
        3
    )

    # Small amount of traffic per attempt
    event.packets_out = random.randint(
        3,
        15
    )

    event.packets_in = random.randint(
        0,
        8
    )

    event.bytes_out = random.randint(
        200,
        4000
    )

    event.bytes_in = random.randint(
        0,
        2500
    )

    event.tcp_flags = random.choice([
        "S",
        "S",
        "PA",
        "R"
    ])

    event.dns_query = None
    event.dns_entropy = None

    event.http_method = None
    event.http_status = None

    return event