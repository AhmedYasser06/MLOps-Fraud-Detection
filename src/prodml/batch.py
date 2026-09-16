"""
Session 3 · Deliverable 03 — Batch scoring, chunked for 1M+ row runs.

Used for the workloads that don't need a live endpoint: e.g. re-scoring a
day's settled transactions for a fraud-analyst review queue, or backfilling
scores after a new model is promoted. Not the path for real-time card
authorization -- that's the streaming consumer in streaming/consumer.py,
because fraud decisions at swipe-time can't wait for a nightly job.

Run:
    python src/prodml/batch.py \
        --input data/scoring/input/2026-09-12.parquet \
        --output data/scoring/output/2026-09-12.parquet \
        --chunk-size 100000

Generate a >=1M row synthetic input to reproduce the handbook's scale
requirement if your real dataset is smaller:
    python src/prodml/batch.py --make-sample data/scoring/input/sample_1m.parquet --rows 1200000
"""

import argparse
import time
import tracemalloc
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.predict_registry import load_production_model

FEATURE_COLUMNS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
)  # matches configs/config.yml -> dataset.train_feature

# Cost assumption used for the cost-per-million estimate below; override
# with your actual instance's hourly price via --instance-cost-per-hour.
DEFAULT_INSTANCE_COST_PER_HOUR = 0.096  # e.g. an AWS m6i.large on-demand rate


def make_sample(path: str, rows: int) -> None:
    """Generates a shape-correct synthetic dataset for exercising batch
    scoring at the >=1M row scale the handbook asks for, without needing
    a real multi-GB export."""
    rng = np.random.default_rng(42)
    data = {"Time": rng.uniform(0, 172792, rows)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1.5, rows)
    data["Amount"] = rng.uniform(1, 2000, rows)
    df = pd.DataFrame(data)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Wrote {rows:,} synthetic rows -> {path}")


def batch_score(
    input_path: str,
    output_path: str,
    chunk_size: int = 100_000,
    instance_cost_per_hour: float = DEFAULT_INSTANCE_COST_PER_HOUR,
) -> None:
    # Always load from the Production stage in the registry.
    model = load_production_model(
        model_name="fraud-detector",
        stage="Production",
    )

    import pyarrow as pa
    import pyarrow.parquet as pq

    # Read row count from Parquet metadata instead of loading the dataset.
    input_pf = pq.ParquetFile(input_path)
    total_rows = input_pf.metadata.num_rows

    print(
        f"Scoring {total_rows:,} transactions "
        f"in chunks of {chunk_size:,} ..."
    )

    tracemalloc.start()
    start = time.perf_counter()

    run_date = date.today().isoformat()

    # Partition output by scoring date:
    #
    # output/
    #   run_date=2026-09-15/
    #       part-00000.parquet
    #
    output_dir = Path(output_path)

    # If --output ends with .parquet, use its parent as the dataset root.
    if output_dir.suffix == ".parquet":
        output_root = output_dir.parent
    else:
        output_root = output_dir

    partition_dir = output_root / f"run_date={run_date}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    output_file = partition_dir / "part-00000.parquet"

    writer = None
    flagged_total = 0
    rows_written = 0

    # Chunked read -> score -> write.
    # Memory usage scales with chunk_size rather than total dataset size.
    for batch in input_pf.iter_batches(
        batch_size=chunk_size,
        columns=FEATURE_COLUMNS,
    ):
        chunk = batch.to_pandas()

        X = chunk[FEATURE_COLUMNS].values
        result = model.predict(X)

        chunk["fraud_probability"] = result["probability"]
        chunk["fraud_prediction"] = result["prediction"]
        chunk["decision_threshold"] = result["threshold"]

        flagged_total += int(sum(result["prediction"]))
        rows_written += len(chunk)

        table = pa.Table.from_pandas(
            chunk,
            preserve_index=False,
        )

        if writer is None:
            writer = pq.ParquetWriter(
                str(output_file),
                table.schema,
            )

        writer.write_table(table)

    if writer is not None:
        writer.close()

    elapsed_s = time.perf_counter() - start

    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rows_per_sec = (
        total_rows / elapsed_s
        if elapsed_s > 0
        else float("inf")
    )

    instance_hours_per_million = (
        (1_000_000 / rows_per_sec) / 3600
        if rows_per_sec
        else 0
    )

    cost_per_million = (
        instance_hours_per_million * instance_cost_per_hour
    )

    print(f"Done -> {output_file}")
    print(f"  rows scored:          {rows_written:,}")
    print(f"  flagged as fraud:     {flagged_total:,}")
    print(f"  wall-clock time:      {elapsed_s:.1f}s")
    print(f"  throughput:           {rows_per_sec:,.0f} rows/sec")
    print(
        f"  peak memory (traced): "
        f"{peak_bytes / 1e6:.1f} MB"
    )
    print(
        f"  est. cost / 1M rows: "
        f"${cost_per_million:.4f} "
        f"(at ${instance_cost_per_hour}/instance-hour)"
    )
    print(f"  output partition:     run_date={run_date}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--instance-cost-per-hour", type=float, default=DEFAULT_INSTANCE_COST_PER_HOUR)
    parser.add_argument("--make-sample", metavar="PATH", help="write a synthetic dataset instead of scoring")
    parser.add_argument("--rows", type=int, default=1_200_000, help="rows for --make-sample")
    args = parser.parse_args()

    if args.make_sample:
        make_sample(args.make_sample, args.rows)
    else:
        if not args.input or not args.output:
            parser.error("--input and --output are required unless using --make-sample")
        batch_score(args.input, args.output, args.chunk_size, args.instance_cost_per_hour)
