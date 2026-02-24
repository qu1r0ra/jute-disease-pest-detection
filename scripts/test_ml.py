import argparse

from jute_disease.engines.ml.test import ML_CLASSIFIERS, test_ml
from jute_disease.utils import DEFAULT_SEED


def main() -> None:
    parser = argparse.ArgumentParser(description="Jute Classical ML Evaluation")
    parser.add_argument(
        "--classifier",
        type=str,
        default="rf",
        choices=list(ML_CLASSIFIERS.keys()),
        help="Classical ML classifier to evaluate",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="crafted",
        choices=["crafted", "raw"],
        help="Type of features used (crafted or raw)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )

    args = parser.parse_args()
    test_ml(
        classifier=args.classifier,
        feature_type=args.feature_type,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
