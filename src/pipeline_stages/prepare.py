import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.helper_utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yml")
    parser.add_argument("--raw", default="data/split/trainval.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    target = config["dataset"]["target"]
    seed = config["random_seed"]

    df = pd.read_csv(args.raw)

    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df[target], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df[target], random_state=seed
    )

    os.makedirs("data/split", exist_ok=True)
    train_df.to_csv("data/split/train.csv", index=False)
    val_df.to_csv("data/split/val.csv", index=False)
    test_df.to_csv("data/split/test.csv", index=False)

    print(
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"(fraud rate train={train_df[target].mean():.4%})"
    )


if __name__ == "__main__":
    main()
