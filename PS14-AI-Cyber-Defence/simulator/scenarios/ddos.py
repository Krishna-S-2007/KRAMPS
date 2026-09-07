import random


TARGET_ASSET = "OPS-01"


def apply_ddos_profile(event, target_ip):
   
    event.destination_ip = target_ip

    event.destination_port = 443
    event.protocol = "TCP"

    # DDoS-like short repeated flows
    event.duration = round(
        random.uniform(0.01, 0.30),
        3
    )

    # Large packet imbalance
    event.packets_out = random.randint(
        150,
        900
    )

    event.packets_in = random.randint(
        0,
        25
    )

    event.bytes_out = random.randint(
        50000,
        700000
    )

    event.bytes_in = random.randint(
        0,
        25000
    )

    event.tcp_flags = random.choice(
        ["S", "S", "S", "A", "PA"]
    )

    event.dns_query = None
    event.dns_entropy = None

    event.http_method = None
    event.http_status = None

    return event