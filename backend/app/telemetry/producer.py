import json
from kafka import KafkaProducer

from backend.app.core.kafka import (
    KAFKA_BOOTSTRAP_SERVERS,
    KafkaTopics,
)


producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)


def publish_telemetry(event: dict) -> None:
    future = producer.send(
        KafkaTopics.SIMULATION,
        value=event
    )

    metadata = future.get(timeout=10)

    print(
        f"Published → topic={metadata.topic}, "
        f"partition={metadata.partition}, "
        f"offset={metadata.offset}"
    )


def close_producer():
    producer.flush()
    producer.close()