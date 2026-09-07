import json
from kafka import KafkaConsumer
from backend.app.telemetry.storage import store_raw_event
from backend.app.core.kafka import (
    KAFKA_BOOTSTRAP_SERVERS,
    KafkaTopics,
)


consumer = KafkaConsumer(
    KafkaTopics.SIMULATION,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="ps14-telemetry-consumer",
    value_deserializer=lambda value: json.loads(value.decode("utf-8"))
)


def consume_telemetry():
    print("PS14 Telemetry Consumer started...")
    print("Waiting for events...\n")

    for message in consumer:
        event = message.value
        print("TELEMETRY RECEIVED")
        print("------------------")
        print(json.dumps(event, indent=2))
        print()
        store_raw_event(event)


if __name__ == "__main__":
    consume_telemetry()