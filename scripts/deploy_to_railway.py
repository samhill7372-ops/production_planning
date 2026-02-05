#!/usr/bin/env python
"""
Deployment helper script for Railway.

This script helps prepare and validate the deployment before pushing to Railway.

Usage:
    python scripts/deploy_to_railway.py --validate
    python scripts/deploy_to_railway.py --prepare <model_name>
"""

import sys
import os
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def validate_model(model_name: str) -> bool:
    """Validate that a model is ready for deployment."""
    model_dir = PROJECT_ROOT / "models" / model_name

    required_files = [
        "yield_model.onnx",
        "encoders.joblib",
        "feature_columns.joblib",
        "historical_data_precomputed.joblib"
    ]

    print(f"\nValidating model: {model_name}")

    if not model_dir.exists():
        print(f"  ERROR: Model directory not found: {model_dir}")
        return False

    all_valid = True
    for filename in required_files:
        filepath = model_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  OK: {filename} ({size_kb:.1f} KB)")
        else:
            print(f"  MISSING: {filename}")
            all_valid = False

    return all_valid


def validate_all() -> dict:
    """Validate all models."""
    models_dir = PROJECT_ROOT / "models"

    if not models_dir.exists():
        print("ERROR: models/ directory not found")
        return {}

    results = {}
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            results[model_name] = validate_model(model_name)

    return results


def copy_models_to_deployment():
    """Copy models to deployment folder."""
    src = PROJECT_ROOT / "models"
    dst = PROJECT_ROOT / "deployment" / "models"

    print(f"\nCopying models from {src} to {dst}")

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    print("  Done!")


def main():
    parser = argparse.ArgumentParser(description="Railway Deployment Helper")
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate all models are ready for deployment'
    )
    parser.add_argument(
        '--prepare', '-p',
        action='store_true',
        help='Prepare deployment (copy models)'
    )
    parser.add_argument(
        'model',
        nargs='?',
        help='Specific model to validate/prepare'
    )

    args = parser.parse_args()

    if args.validate:
        if args.model:
            valid = validate_model(args.model)
            sys.exit(0 if valid else 1)
        else:
            results = validate_all()
            print("\n" + "=" * 50)
            print("VALIDATION SUMMARY")
            print("=" * 50)
            all_valid = True
            for name, valid in results.items():
                status = "READY" if valid else "NOT READY"
                print(f"  {name}: {status}")
                if not valid:
                    all_valid = False

            sys.exit(0 if all_valid else 1)

    elif args.prepare:
        # Validate first
        results = validate_all()
        all_valid = all(results.values())

        if not all_valid:
            print("\nERROR: Some models are not valid. Fix errors before deploying.")
            sys.exit(1)

        # Copy models
        copy_models_to_deployment()

        print("\n" + "=" * 50)
        print("DEPLOYMENT READY")
        print("=" * 50)
        print("\nNext steps:")
        print("  1. git add deployment/")
        print("  2. git commit -m 'Update models for deployment'")
        print("  3. git push origin main")
        print("  4. Railway will auto-deploy")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
