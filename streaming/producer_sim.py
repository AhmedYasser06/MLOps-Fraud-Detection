"""
Session 3 · supporting script — simulate incoming transactions onto the
Redis Stream, so streaming/consumer.py has something to consume without
needing a real payment gateway hooked up.

Run:
    python streaming/producer_sim.py --rate 100 --duration 30
"""

import argparse
import json
import random
import time
import uuid

import redis


def random_transaction() -> dict:
    payload = {
    "transaction_id": str(uuid.uuid4()),
    "event_created_at": time.time(),
    "Time": time.time(),
    }

    for i in range(1, 29):
        payload[f"V{i}"] = round(random.gauss(0, 1), 4)
    payload["Amount"] = round(random.uniform(1, 2000), 2)
    return payload


def main(rate: int, duration: int, host: str) -> None:
    r = redis.Redis(host=host, decode_responses=True)
    interval = 1.0 / rate
    end = time.time() + duration
    sent = 0

    while time.time() < end:
        payload = random_transaction()
        r.xadd("transactions", {"data": json.dumps(payload)})
        sent += 1
        time.sleep(interval)

    print(f"Sent {sent} simulated transactions at ~{rate}/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=100, help="events per second")
    parser.add_argument("--duration", type=int, default=30, help="seconds to run")
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()
    main(args.rate, args.duration, args.host)
