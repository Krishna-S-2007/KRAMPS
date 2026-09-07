KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"


class KafkaTopics:
    SIMULATION = "ps14.raw.simulation"

    WINDOWS = "ps14.raw.windows"
    LINUX = "ps14.raw.linux"
    NETWORK = "ps14.raw.network"
    IDS = "ps14.raw.ids"
    NETFLOW = "ps14.raw.netflow"
    THREAT_INTEL = "ps14.raw.threatintel"

    CANONICAL = "ps14.canonical.events"
    ALERTS = "ps14.alerts"
    INCIDENTS = "ps14.incidents"
    PROGRESSION = "ps14.progression"
    RISK = "ps14.risk"
    PREDICTIONS = "ps14.predictions"
    ACTIONS = "ps14.actions"