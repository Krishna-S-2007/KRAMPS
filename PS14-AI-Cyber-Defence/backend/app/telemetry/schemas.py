from datetime import datetime

from pydantic import BaseModel


class TelemetryEvent(BaseModel):
    timestamp: datetime

    source_ip: str
    destination_ip: str

    source_port: int
    destination_port: int

    protocol: str

    duration: float

    bytes_in: int
    bytes_out: int

    packets_in: int
    packets_out: int

    tcp_flags: str | None = None

    dns_query: str | None = None
    dns_entropy: float | None = None

    http_method: str | None = None
    http_status: int | None = None