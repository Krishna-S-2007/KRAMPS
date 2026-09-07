import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def load_assets():
    path = BASE_DIR / "assets.json"

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_topology():
    path = BASE_DIR / "topology.json"

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_asset(asset_id: str):
    assets = load_assets()

    for asset in assets:
        if asset["asset_id"] == asset_id:
            return asset

    return None


if __name__ == "__main__":
    assets = load_assets()
    topology = load_topology()

    print("PS14 SIMULATED NETWORK")
    print("----------------------")
    print(f"Assets: {len(assets)}")
    print(f"Connections: {len(topology['connections'])}")

    for asset in assets:
        print(
            f"{asset['asset_id']} | "
            f"{asset['ip']} | "
            f"{asset['role']} | "
            f"criticality={asset['criticality']}"
        )