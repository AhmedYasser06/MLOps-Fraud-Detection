#!/usr/bin/env python3

import math
import subprocess
import time


NGINX_CONTAINER = "fraud-cat9-nginx"
NGINX_LOG = "/var/log/nginx/cat9_access.log"

INTERVAL_SECONDS = 10

P95_THRESHOLD_MS = 300.0
ERROR_RATE_THRESHOLD = 0.05

REQUIRED_BAD_INTERVALS = 2


def get_log():
    result = subprocess.run(
        [
            "docker",
            "exec",
            NGINX_CONTAINER,
            "cat",
            NGINX_LOG,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.splitlines()


def parse_line(line):
    parts = line.strip().split()

    deployment = None
    status = None
    upstream_time = None

    for part in parts:
        if part.startswith("deployment="):
            deployment = part.split("=", 1)[1]

        elif part.startswith("status="):
            status = int(part.split("=", 1)[1])

        elif part.startswith("upstream_time="):
            value = part.split("=", 1)[1]

            if value != "-":
                upstream_time = float(value)

    if deployment != "canary":
        return None

    if upstream_time is None:
        return None

    return {
        "status": status,
        "latency_ms": upstream_time * 1000.0,
    }


def percentile(values, p):
    values = sorted(values)

    if not values:
        return None

    rank = (p / 100.0) * (len(values) - 1)

    lower = math.floor(rank)
    upper = math.ceil(rank)

    if lower == upper:
        return values[lower]

    weight = rank - lower

    return values[lower] * (1 - weight) + values[upper] * weight


def rollback():
    print()
    print("!!! CANARY UNHEALTHY — EXECUTING ROLLBACK !!!")

    subprocess.run(
        ["./scripts/cat9/rollback.sh"],
        check=True,
    )


def main():

    print("=" * 60)
    print("CAT9 CANARY WATCHER")
    print("=" * 60)
    print(f"p95 threshold:          {P95_THRESHOLD_MS:.0f} ms")
    print(f"error threshold:        " f"{ERROR_RATE_THRESHOLD * 100:.1f}%")
    print(f"required bad intervals: " f"{REQUIRED_BAD_INTERVALS}")
    print(f"interval:                {INTERVAL_SECONDS}s")
    print("=" * 60)

    # Snapshot the existing log.
    # Anything before this point is ignored.
    previous_lines = get_log()
    previous_count = len(previous_lines)

    bad_intervals = 0

    while True:

        time.sleep(INTERVAL_SECONDS)

        current_lines = get_log()

        # Handle normal append-only growth.
        if len(current_lines) >= previous_count:
            new_lines = current_lines[previous_count:]
        else:
            # Log was rotated/truncated.
            new_lines = current_lines

        previous_count = len(current_lines)

        requests = []

        for line in new_lines:
            parsed = parse_line(line)

            if parsed is not None:
                requests.append(parsed)

        if not requests:
            print("No new canary requests in this interval.")
            continue

        latencies = [request["latency_ms"] for request in requests]

        errors = [request for request in requests if request["status"] >= 500]

        p95 = percentile(latencies, 95)
        error_rate = len(errors) / len(requests)

        unhealthy = p95 > P95_THRESHOLD_MS or error_rate > ERROR_RATE_THRESHOLD

        status = "BAD" if unhealthy else "HEALTHY"

        print(
            f"canary_requests={len(requests)} "
            f"p95={p95:.2f}ms "
            f"errors={len(errors)} "
            f"error_rate={error_rate * 100:.2f}% "
            f"status={status}"
        )

        if unhealthy:

            bad_intervals += 1

            print(f"Bad interval " f"{bad_intervals}/{REQUIRED_BAD_INTERVALS}")

            if bad_intervals >= REQUIRED_BAD_INTERVALS:

                rollback()

                print()
                print("Watcher stopping after rollback.")
                break

        else:

            if bad_intervals > 0:
                print("Canary recovered; resetting bad interval counter.")

            bad_intervals = 0


if __name__ == "__main__":
    main()
