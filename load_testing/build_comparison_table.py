"""
Session 3 · Deliverable 08 (report artifact) — turn several Locust CSV runs
into the exact comparison table the handbook's Step 8 asks for.

Run the same locustfile.py headless against each serving tier first,
each into its own --csv prefix:

    locust -f load_testing/locustfile.py --host http://localhost:8000 \
        --users 100 --spawn-rate 10 --run-time 3m --headless \
        --csv=results/cat1_fastapi
    locust -f load_testing/locustfile.py --host http://localhost:8004 \
        --users 100 --spawn-rate 10 --run-time 3m --headless \
        --csv=results/cat2_bentoml
    locust -f load_testing/locustfile.py --host http://localhost:8005 \
        --users 100 --spawn-rate 10 --run-time 3m --headless \
        --csv=results/cat4_onnxruntime

Then:
    python load_testing/build_comparison_table.py \
        --run "CAT 1,FastAPI eager,results/cat1_fastapi" \
        --run "CAT 2,BentoML + micro-batching,results/cat2_bentoml" \
        --run "CAT 4,ONNX Runtime,results/cat4_onnxruntime"

Prints a Markdown table -- paste it directly into reports/module-3.md.
"""

import argparse
import csv


def read_stats(csv_prefix: str) -> dict:
    """Locust's *_stats.csv has one row per endpoint plus an 'Aggregated'
    row -- that aggregated row is what the handbook's table wants."""
    with open(f"{csv_prefix}_stats.csv") as f:
        rows = list(csv.DictReader(f))
    agg = next(r for r in rows if r["Name"] == "Aggregated")
    total = int(agg["Request Count"])
    failures = int(agg["Failure Count"])
    return {
        "p50": agg.get("50%", agg.get("Median Response Time", "?")),
        "p95": agg.get("95%", "?"),
        "p99": agg.get("99%", "?"),
        "rps": agg.get("Requests/s", "?"),
        "failure_pct": f"{(failures / total * 100):.2f}%" if total else "0.00%",
    }


def main(runs: list[str]) -> None:
    print("| CATEGORY | RUNTIME | P50 | P95 | P99 | RPS @100 | FAILURE % |")
    print("|---|---|---|---|---|---|---|")
    for entry in runs:
        category, runtime, csv_prefix = entry.split(",", 2)
        stats = read_stats(csv_prefix)
        print(
            f"| {category} | {runtime} | {stats['p50']}ms | {stats['p95']}ms | "
            f"{stats['p99']}ms | {stats['rps']} | {stats['failure_pct']} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", required=True,
        help='repeatable: "CATEGORY,RUNTIME NAME,path/to/csv_prefix"',
    )
    args = parser.parse_args()
    main(args.run)
