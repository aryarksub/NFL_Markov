import json
from pathlib import Path
import argparse
import pandas as pd

from src import util

ALL_METRICS_CSV = Path(util.METRICS_DIR) / "all_metrics.csv"

def json_metrics_to_csv(
    directory: str | Path,
    exclude_strings: list[str],
    output_path: str | Path,
) -> None:
    """
    Recursively find JSON files in a directory, flatten their metrics into rows,
    and write the resulting DataFrame to a CSV file.

    Args:
        directory: Root directory to recursively search for JSON files.
        exclude_strings: Skip any JSON file whose filename contains any of these strings.
        output_path: Path where the output CSV should be written.
    """
    directory = Path(directory)
    output_path = Path(output_path)

    rows = []

    # rglob recursively searches all subdirectories.
    for json_file in directory.rglob("*.json"):
        # Skip files whose name contains any excluded string.
        if any(exclude_string in json_file.name for exclude_string in exclude_strings):
            continue

        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        row = {
            "model": data.get("model"),
            "sim_mode": data.get("sim_mode"),
            "split_mode": data.get("split_mode"),
            "combined_name": data.get("combined_name"),
            **data["metrics"],
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Create the parent directory if necessary.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Wrote {len(df)} rows to {output_path}")

def main(
    directory,
    exclude_strings,
    output_path
):
    if str(output_path) == str(ALL_METRICS_CSV):
        response = input(
            f"Are you sure you want to overwrite the output at the default path {ALL_METRICS_CSV}? [Y/N]: "
        ).strip().lower()

        if response != "y":
            print("Operation cancelled. Please rerun the program.")
            return

    json_metrics_to_csv(
        directory=directory,
        exclude_strings=exclude_strings,
        output_path=output_path,
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Recursively combine JSON model metrics into a CSV."
    )

    parser.add_argument(
        "--directory", "-dir",
        type=Path,
        default=util.METRICS_DIR,
        help="Directory to recursively search for JSON files.",
    )

    parser.add_argument(
        "--output-path", "--out-path", "-out",
        type=Path,
        default=ALL_METRICS_CSV,
        help="Path for the output CSV file.",
    )

    parser.add_argument(
        "--exclude-strings", "--exclude", "-excl",
        nargs="*",
        default=[],
        help=(
            "Strings to exclude. Any JSON file whose filename contains one of these strings will be skipped."
        ),
    )

    args = parser.parse_args()
    return args    

if __name__ == "__main__":
    args = parse_args()

    main(
        directory=args.directory,
        exclude_strings=args.exclude_strings,
        output_path=args.output_path
    )