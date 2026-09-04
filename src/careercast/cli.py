"""Command-line interface for CareerCast."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

from .client import CareerCastAPIError, CareerCastClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CareerCast API command-line client")
    parser.add_argument(
        "--api-url",
        default=os.getenv("CAREERCAST_API_URL", "http://127.0.0.1:8000"),
        help="CareerCast API base URL",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health", help="Check API and model availability")
    subparsers.add_parser("models", help="Show registered model information")

    predict = subparsers.add_parser("predict", help="Predict likely careers")
    predict.add_argument("skills_text")
    predict.add_argument("--top-k", type=int, default=5)

    recommend = subparsers.add_parser("recommend", help="Get ensemble recommendations")
    recommend.add_argument("skills_text")
    recommend.add_argument("--top-k", type=int, default=10)

    gap = subparsers.add_parser("gap", help="Generate a skill-gap report")
    gap.add_argument("skills_text")
    gap.add_argument("--target-career")
    gap.add_argument("--top-k-careers", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = CareerCastClient(args.api_url)
    try:
        if args.command == "health":
            result = client.health()
        elif args.command == "models":
            result = client.models_info()
        elif args.command == "predict":
            result = client.predict(args.skills_text, top_k=args.top_k)
        elif args.command == "recommend":
            result = client.recommend(args.skills_text, top_k=args.top_k)
        else:
            result = client.gap_report(
                args.skills_text,
                target_career=args.target_career,
                top_k_careers=args.top_k_careers,
            )
    except CareerCastAPIError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
