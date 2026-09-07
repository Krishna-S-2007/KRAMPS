import random


TARGET_ASSET = "COMM-01"


def apply_communication_disruption_profile(
    event,
    target_asset
):
    event.destination_ip = target_asset["ip"]

    event.destination_port = 443
    event.protocol = "TCP"

    # Very high delay
    event.duration = round(
        random.uniform(8.0, 30.0),
        3
    )

    # Very little successful response
    event.packets_out = random.randint(
        10,
        80
    )

    event.packets_in = random.randint(
        0,
        5
    )

    event.bytes_out = random.randint(
        2000,
        40000
    )

    event.bytes_in = random.randint(
        0,
        3000
    )

    event.tcp_flags = random.choice([
        "S",
        "R",
        "FA"
    ])

    event.dns_query = None
    event.dns_entropy = None

    event.http_method = None
    event.http_status = None

    return event