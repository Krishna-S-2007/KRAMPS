import random
import string


SOURCE_ASSET = "HOST-07"
DNS_ASSET = "DNS-01"


def generate_high_entropy_domain():
    random_part = "".join(
        random.choices(
            string.ascii_lowercase +
            string.digits,
            k=random.randint(16, 28)
        )
    )

    return f"{random_part}.internal-update.local"


def apply_dns_c2_profile(
    event,
    source_asset,
    dns_asset
):
    event.source_ip = source_asset["ip"]
    event.destination_ip = dns_asset["ip"]

    event.source_port = random.randint(
        49152,
        65535
    )

    event.destination_port = 53
    event.protocol = "UDP"

    event.duration = round(
        random.uniform(0.01, 0.15),
        3
    )

    event.bytes_out = random.randint(80, 400)
    event.bytes_in = random.randint(50, 300)

    event.packets_out = random.randint(1, 4)
    event.packets_in = random.randint(1, 4)

    event.tcp_flags = None

    event.dns_query = generate_high_entropy_domain()

    event.dns_entropy = round(
        random.uniform(3.8, 5.5),
        2
    )

    event.http_method = None
    event.http_status = None

    return event