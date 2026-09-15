"""Prometheus metrics for the urgency triage API."""

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "predict_requests_total",
    "Total number of /predict requests",
    labelnames=["status"],
)

REQUEST_DURATION = Histogram(
    "predict_request_duration_seconds",
    "Duration of /predict requests in seconds",
)
