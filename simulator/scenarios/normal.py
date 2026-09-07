import random


def apply_normal_profile(event):
    
    event.duration = round(
        random.uniform(0.2, 5.0),
        3
    )

    event.packets_out = random.randint(3, 40)
    event.packets_in = random.randint(2, 30)

    event.bytes_out = random.randint(500, 50000)
    event.bytes_in = random.randint(300, 30000)

    if event.protocol == "TCP":
        event.tcp_flags = random.choice(
            ["A", "PA", "FA"]
        )

    return event