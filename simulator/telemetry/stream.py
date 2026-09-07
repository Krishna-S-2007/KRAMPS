import argparse
import random
import time

from datetime import datetime, timezone

from backend.app.telemetry.schemas import TelemetryEvent
from backend.app.telemetry.producer import publish_telemetry

from simulator.network.loader import (
    load_assets,
    load_topology
)

from simulator.scenarios.normal import (
    apply_normal_profile
)

from simulator.scenarios.ddos import (
    apply_ddos_profile,
    TARGET_ASSET as DDOS_TARGET
)

from simulator.scenarios.reconnaissance import (
    apply_recon_profile
)

from simulator.scenarios.brute_force import (
    apply_brute_force_profile,
    SOURCE_ASSET as BRUTE_SOURCE,
    TARGET_ASSET as BRUTE_TARGET
)

from simulator.scenarios.lateral_movement import (
    apply_lateral_movement_profile,
    SOURCE_ASSET as LATERAL_SOURCE,
    LATERAL_TARGETS
)

from simulator.scenarios.exfiltration import (
    apply_exfiltration_profile,
    SOURCE_ASSET as EXFIL_SOURCE
)

from simulator.scenarios.dns_c2 import (
    apply_dns_c2_profile,
    SOURCE_ASSET as DNS_C2_SOURCE,
    DNS_ASSET
)

from simulator.scenarios.communication_disruption import (
    apply_communication_disruption_profile,
    TARGET_ASSET as COMM_TARGET
)


# --------------------------------------------------
# LOAD NETWORK
# --------------------------------------------------

assets = load_assets()
topology = load_topology()

asset_map = {
    asset["asset_id"]: asset
    for asset in assets
}


# --------------------------------------------------
# RANDOM SCENARIO DISTRIBUTION
# --------------------------------------------------

SCENARIO_WEIGHTS = {
    "normal": 75,
    "reconnaissance": 6,
    "brute_force": 5,
    "ddos": 4,
    "lateral_movement": 3,
    "exfiltration": 2,
    "dns_c2": 3,
    "communication_disruption": 2
}


def choose_scenario():

    scenarios = list(
        SCENARIO_WEIGHTS.keys()
    )

    weights = list(
        SCENARIO_WEIGHTS.values()
    )

    return random.choices(
        scenarios,
        weights=weights,
        k=1
    )[0]


# --------------------------------------------------
# BASE EVENT GENERATOR
# --------------------------------------------------

def generate_base_event():

    # Choose a valid normal connection
    connection = random.choice(
        topology["connections"]
    )

    source = asset_map[
        connection["source"]
    ]

    destination = asset_map[
        connection["destination"]
    ]

    protocol = random.choice(
        connection["protocols"]
    )

    destination_port = random.choice(
        connection["ports"]
    )

    tcp_flags = None
    dns_query = None
    dns_entropy = None
    http_method = None
    http_status = None


    # TCP
    if protocol == "TCP":

        tcp_flags = random.choice([
            "A",
            "PA",
            "FA"
        ])


    # DNS
    if destination_port == 53:

        dns_query = random.choice([
            "ops.internal",
            "command.internal",
            "files.internal",
            "communications.internal"
        ])

        dns_entropy = round(
            random.uniform(
                1.5,
                3.0
            ),
            2
        )


    # HTTP / HTTPS
    if destination_port in [80, 443]:

        http_method = random.choice([
            "GET",
            "POST"
        ])

        http_status = random.choice([
            200,
            200,
            200,
            204
        ])


    return TelemetryEvent(

        timestamp=datetime.now(
            timezone.utc
        ),

        source_ip=source["ip"],
        destination_ip=destination["ip"],

        source_port=random.randint(
            49152,
            65535
        ),

        destination_port=destination_port,

        protocol=protocol,

        duration=1.0,

        bytes_in=1000,
        bytes_out=1000,

        packets_in=10,
        packets_out=10,

        tcp_flags=tcp_flags,

        dns_query=dns_query,
        dns_entropy=dns_entropy,

        http_method=http_method,
        http_status=http_status
    )


# --------------------------------------------------
# SCENARIO ENGINE
# --------------------------------------------------

def apply_scenario(event, scenario):

    if scenario == "normal":

        return apply_normal_profile(
            event
        )


    elif scenario == "ddos":

        target = asset_map[
            DDOS_TARGET
        ]

        return apply_ddos_profile(
            event,
            target["ip"]
        )


    elif scenario == "reconnaissance":

        target = asset_map[
            "OPS-01"
        ]

        return apply_recon_profile(
            event,
            target["ip"]
        )


    elif scenario == "brute_force":

        source = asset_map[
            BRUTE_SOURCE
        ]

        target = asset_map[
            BRUTE_TARGET
        ]

        return apply_brute_force_profile(
            event,
            source,
            target
        )


    elif scenario == "lateral_movement":

        source = asset_map[
            LATERAL_SOURCE
        ]

        target_id = random.choice(
            LATERAL_TARGETS
        )

        target = asset_map[
            target_id
        ]

        return apply_lateral_movement_profile(
            event,
            source,
            target
        )


    elif scenario == "exfiltration":

        source = asset_map[
            EXFIL_SOURCE
        ]

        return apply_exfiltration_profile(
            event,
            source
        )


    elif scenario == "dns_c2":

        source = asset_map[
            DNS_C2_SOURCE
        ]

        dns = asset_map[
            DNS_ASSET
        ]

        return apply_dns_c2_profile(
            event,
            source,
            dns
        )


    elif scenario == "communication_disruption":

        target = asset_map[
            COMM_TARGET
        ]

        return apply_communication_disruption_profile(
            event,
            target
        )


    return event


# --------------------------------------------------
# EVENT DELAY
# --------------------------------------------------

def get_delay(scenario):

    if scenario == "ddos":

        return random.uniform(
            0.05,
            0.25
        )


    elif scenario == "reconnaissance":

        return random.uniform(
            0.1,
            0.4
        )


    elif scenario == "brute_force":

        return random.uniform(
            0.1,
            0.5
        )


    elif scenario == "dns_c2":

        return random.uniform(
            0.5,
            1.5
        )


    elif scenario == "lateral_movement":

        return random.uniform(
            0.5,
            1.2
        )


    elif scenario == "exfiltration":

        return random.uniform(
            1.0,
            2.5
        )


    elif scenario == "communication_disruption":

        return random.uniform(
            0.8,
            2.0
        )


    return random.uniform(
        0.5,
        1.5
    )


# --------------------------------------------------
# LIVE STREAM
# --------------------------------------------------

def start_stream(forced_scenario=None):

    print()
    print("=" * 65)
    print("PS14 LIVE DEFENCE NETWORK")
    print("=" * 65)

    print(
        f"Assets loaded : {len(assets)}"
    )

    if forced_scenario:

        print(
            f"Mode          : FORCED {forced_scenario.upper()}"
        )

    else:

        print(
            "Mode          : LIVE RANDOM"
        )

    print(
        "Kafka topic   : ps14.raw.simulation"
    )

    print()
    print("Press CTRL+C to stop.")
    print()


    try:

        while True:

            # Generate valid base telemetry
            event = generate_base_event()


            # Manual scenario if supplied
            if forced_scenario:

                scenario = forced_scenario

            else:

                scenario = choose_scenario()


            # Apply behaviour profile
            event = apply_scenario(
                event,
                scenario
            )


            # Publish to Kafka
            publish_telemetry(
                event.model_dump(
                    mode="json"
                )
            )


            print(
                f"[{scenario.upper():<24}] "
                f"{event.source_ip:<15} "
                f"→ "
                f"{event.destination_ip:<15} "
                f":{event.destination_port:<5} "
                f"{event.protocol:<4} "
                f"pkts={event.packets_out:<5} "
                f"bytes={event.bytes_out}"
            )


            time.sleep(
                get_delay(
                    scenario
                )
            )


    except KeyboardInterrupt:

        print()
        print(
            "PS14 network simulator stopped."
        )


# --------------------------------------------------
# COMMAND LINE
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="PS14 Network Simulator"
    )

    parser.add_argument(
        "--scenario",

        choices=[
            "normal",
            "ddos",
            "reconnaissance",
            "brute_force",
            "lateral_movement",
            "exfiltration",
            "dns_c2",
            "communication_disruption"
        ],

        default=None,

        help=(
            "Force a scenario. "
            "If omitted, the simulator "
            "generates a mixed live stream."
        )
    )


    args = parser.parse_args()

    start_stream(
        args.scenario
    )