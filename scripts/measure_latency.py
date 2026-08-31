"""Measures baseline latency of the /predict endpoint.

Requires the API to already be running locally (uvicorn) before executing
this script. Sends repeated requests and reports mean, median, and p95
latency, in milliseconds.
"""

import statistics
import time

import httpx
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"
SAMPLE_TEXT = "Patient presents with severe chest pain and shortness of breath."
NUM_REQUESTS = 50


def measure_single_request(text: str) -> float:
    """Times a single POST /predict request.

    Args:
        text: The text to send in the request body.

    Returns:
        Elapsed time, in seconds.
    """
    start = time.perf_counter()
    httpx.post(API_URL, json={"text": text})
    end = time.perf_counter()
    return end - start


def main() -> None:
    """Runs repeated requests and reports latency statistics."""
    durations = []
    for i in range(NUM_REQUESTS):
        duration = measure_single_request(SAMPLE_TEXT)
        durations.append(duration)
        print(f"Request {i + 1}: {duration * 1000:.2f} ms")

    mean = statistics.mean(durations)
    median = statistics.median(durations)
    p95 = np.percentile(durations, 95)

    print()
    print(f"Mean:   {mean * 1000:.2f} ms")
    print(f"Median: {median * 1000:.2f} ms")
    print(f"P95:    {p95 * 1000:.2f} ms")


if __name__ == "__main__":
    main()
