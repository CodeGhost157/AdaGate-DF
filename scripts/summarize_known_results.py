#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from deepfake_lowres.results.known_results import NOTEBOOK_RESULTS


def main():
    rows = []
    for key, result in NOTEBOOK_RESULTS.items():
        rows.append({
            "id": key,
            "model": result["model"],
            "dataset": result["dataset"],
            "accuracy": result.get("accuracy"),
            "auc": result.get("auc"),
            "f1": result.get("f1"),
            "training_minutes": result.get("training_minutes"),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
