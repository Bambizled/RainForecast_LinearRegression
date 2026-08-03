import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data.preprocessing import main as run_preprocessing
from src.features.feature_engineering import main as run_feature_engineering
from src.models.train import main as run_training
from src.visualization.plots import main as run_visualization


def main() -> None:
    print("Running Preprocessing Pipeline...")
    run_preprocessing()

    print("Running Feature Selection Pipeline...")
    run_feature_engineering()

    print("Running Model Training and Evaluation...")
    run_training()

    print("Generating Figures and Visualizations...")
    run_visualization()

    print("Pipeline Execution Completed Successfully!")


if __name__ == "__main__":
    main()
