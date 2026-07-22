import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data.preprocessing import main as run_preprocessing
from src.features.feature_engineering import main as run_feature_engineering


def main() -> None:
    print("=" * 80)
    print("     running preprocessing and feature selection pipeline")
    print("=" * 80)

    print("\n--- Step 1: Preprocessing data ---")
    run_preprocessing()

    print("\n--- Step 2: Feature selection ---")
    run_feature_engineering()

    print("\n" + "=" * 80)
    print("     pipeline execution completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
