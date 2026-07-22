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
    print("      RUNNING PREPROCESSING AND FEATURE SELECTION PIPELINE")
    print("=" * 80)

    print("\n--- STEP 1: PREPROCESSING DATA ---")
    run_preprocessing()

    print("\n--- STEP 2: FEATURE SELECTION ---")
    run_feature_engineering()

    print("\n" + "=" * 80)
    print("      PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
