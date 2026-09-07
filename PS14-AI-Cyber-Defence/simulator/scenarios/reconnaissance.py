import random


def apply_recon_profile(event, target_ip):
    event.destination_ip = target_ip

    event.destination_port = random.choice([
        21, 22, 23, 25, 53, 80, 135,
        139, 443, 445, 3389, 8080
    ])

    event.protocol = "TCP"

    event.duration = round(
        random.uniform(0.01, 0.20),
        3
    )

    event.packets_out = random.randint(1, 8)
    event.packets_in = random.randint(0, 4)

    event.bytes_out = random.randint(40, 1000)
    event.bytes_in = random.randint(0, 700)

    event.tcp_flags = random.choice([
        "S",
        "S",
        "S",
        "R"
    ])

    event.dns_query = None
    event.dns_entropy = None

    event.http_method = None
    event.http_status = None

    return event