from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import yaml
import os


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIMENSIONS = (
    "model",
    "sim_mode",
    "split_mode",
)

IDENTIFIER_COLUMNS = set(DIMENSIONS)

# Optional metadata file. This can be changed with --model-metadata.
DEFAULT_MODEL_METADATA = "model_metadata.yaml"

DEFAULT_METRICS_FILE = os.path.join('metrics', 'all_metrics.csv')

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load the evaluation dataframe from CSV."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataframe CSV does not exist: {path}")

    return pd.read_csv(path)


def load_model_metadata(path: str | Path) -> dict:
    """Load model metadata from YAML."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model metadata YAML does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data.get("models", {})


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def get_metric_columns(df: pd.DataFrame) -> list[str]:
    """
    Return columns that appear to be metric columns.

    A metric is currently defined as a numeric column that is not one of
    the known identifier/dimension columns.
    """
    metrics = []

    for column in df.columns:
        if column in IDENTIFIER_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            metrics.append(column)

    return metrics


def validate_metrics(
    df: pd.DataFrame,
    metrics: Sequence[str],
) -> None:
    """Validate that all requested metrics exist."""
    available = set(get_metric_columns(df))

    missing = [metric for metric in metrics if metric not in available]

    if missing:
        raise ValueError(
            "Unknown metric(s): "
            + ", ".join(missing)
            + "\n\nAvailable metrics:\n"
            + "\n".join(f"  {metric}" for metric in sorted(available))
        )


# ---------------------------------------------------------------------------
# Dataset description
# ---------------------------------------------------------------------------

def describe_data(df: pd.DataFrame) -> None:
    """
    Print a concise description of the dataset.
    """
    print()
    print("Dataset")
    print("-------")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print()
    print("Dimensions")
    print("----------")

    for dimension in sorted(DIMENSIONS):
        if dimension not in df.columns:
            continue

        values = (
            df[dimension]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        values.sort()

        print(f"\n{dimension} ({len(values)} values):")

        for value in values:
            count = (
                df[dimension]
                .astype("string")
                .eq(value)
                .sum()
            )

            print(f"  {value}: {count:,} rows")

    print()
    print("Metrics")
    print("-------")

    metrics = get_metric_columns(df)

    for metric in metrics:
        dtype = df[metric].dtype
        non_null = df[metric].notna().sum()

        print(
            f"  {metric:<45} "
            f"dtype={str(dtype):<10} "
            f"non-null={non_null:,}"
        )

    print()


def list_metrics(df: pd.DataFrame) -> None:
    """Print available metric columns."""
    metrics = get_metric_columns(df)

    print(f"Available metrics ({len(metrics)}):")
    print()

    for metric in metrics:
        print(f"  {metric}")


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

def get_model_type(
    model_metadata: dict,
    model: str,
) -> str | None:
    """Return the configured model type, if available."""
    metadata = model_metadata.get(model, {})
    return metadata.get("type")


def model_has_sim_mode(
    df: pd.DataFrame,
    model_metadata: dict,
    model: str,
) -> bool:
    """
    Determine whether a model should be considered to have simulation modes.

    The YAML metadata is authoritative when a model is defined there.

    For backwards compatibility, if the model is not present in YAML,
    infer this from whether the model has any non-null sim_mode values.
    """
    if model in model_metadata:
        model_type = get_model_type(model_metadata, model)

        # Regression models do not have sim modes.
        if model_type == "regression":
            return False

        # Markov/classification-style models generally do.
        if model_type == "markov":
            return True

    model_rows = df[df["model"].eq(model)]

    return model_rows["sim_mode"].notna().any()


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def normalize_filter(
    values: str | Sequence[str] | None,
) -> list[str] | None:
    """
    Normalize a filter into a list.

    Supports:
        None
        "simple_markov"
        "simple_markov,lin_reg"
        ["simple_markov", "lin_reg"]
    """
    if values is None:
        return None

    if isinstance(values, str):
        values = values.split(",")

    result = [
        str(value).strip()
        for value in values
        if str(value).strip()
    ]

    return result or None


def filter_dataframe(
    df: pd.DataFrame,
    model_metadata: dict,
    models: Sequence[str] | None = None,
    sim_modes: Sequence[str] | None = None,
    split_modes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Filter the dataframe according to model/sim_mode/split_mode.

    Important behavior:

    If a model does not have simulation modes, sim_modes does not filter
    that model.

    Example:

        models = ["simple_markov", "lin_reg"]
        sim_modes = ["greedy", "sample1"]

    will retain:

        simple_markov + greedy
        simple_markov + sample1
        lin_reg + all rows

    when lin_reg is a regression model with no sim_mode.
    """
    models = normalize_filter(models)
    sim_modes = normalize_filter(sim_modes)
    split_modes = normalize_filter(split_modes)

    result = df.copy()

    # Model filtering
    if models is not None:
        result = result[result["model"].isin(models)]

    # Split-mode filtering applies uniformly.
    if split_modes is not None:
        result = result[result["split_mode"].isin(split_modes)]

    # Simulation-mode filtering is model-aware.
    if sim_modes is not None:
        keep = pd.Series(False, index=result.index)

        model_values = (
            result["model"]
            .dropna()
            .astype(str)
            .unique()
        )

        for model in model_values:
            model_mask = result["model"].eq(model)

            if model_has_sim_mode(
                df,
                model_metadata,
                model,
            ):
                # Model has simulation modes, so apply the filter.
                keep |= (
                    model_mask
                    & result["sim_mode"].isin(sim_modes)
                )
            else:
                # Model has no simulation modes, so retain all rows.
                keep |= model_mask

        result = result[keep]

    return result.copy()


# ---------------------------------------------------------------------------
# Plot naming
# ---------------------------------------------------------------------------

def make_plot_name(
    models: Sequence[str] | None,
    sim_modes: Sequence[str] | None,
    split_modes: Sequence[str] | None,
    metrics: Sequence[str],
    joint: bool,
) -> str:
    """Create a concise, deterministic plot name."""

    def clean(value: str) -> str:
        """Remove underscores and whitespace from a name."""
        return (
            value
            .replace("_", "")
            .replace(" ", "")
        )

    def dimension(
        values: Sequence[str] | None,
    ) -> str:
        if values is None:
            return "all"

        return "_".join(clean(value) for value in values)

    model_part = dimension(models)
    sim_mode_part = dimension(sim_modes)
    split_mode_part = dimension(split_modes)
    metric_part = "_".join(clean(metric) for metric in metrics)
    plot_type = "jnt" if joint else "sep"

    return "__".join(
        [
            model_part,
            sim_mode_part,
            split_mode_part,
            metric_part,
            plot_type,
        ]
    )


# ---------------------------------------------------------------------------
# Plot sizing
# ---------------------------------------------------------------------------

def calculate_figure_width(
    n_groups: int,
    joint: bool,
) -> float:
    """
    Calculate a reasonable figure width based on the number of groups.

    joint=False:
        one bar per group

    joint=True:
        one group contains one bar per metric
    """
    if n_groups == 0:
        return 8.0

    bars_per_group = 2 if joint else 1

    # Roughly 1.0-1.2 inches per group.
    width = n_groups * 1.15

    if joint:
        width *= 1.05

    return max(8.0, min(width, 30.0))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def get_bar_colors(
    metrics: Sequence[str],
) -> list:
    """Return one color per metric."""
    cmap = plt.get_cmap("tab10")

    return [
        cmap(i % 10)
        for i in range(len(metrics))
    ]


def prepare_plot_data(
    df: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """
    Prepare rows for plotting.

    Rows with no value for any requested metric are removed.
    """
    columns = ["combined_name", *metrics]

    result = df[columns].copy()

    result = result.dropna(
        subset=metrics,
        how="all",
    )

    return result

def plot_metric(
    df: pd.DataFrame,
    metric: str,
    ax: plt.Axes,
    *,
    show_values: bool = True,
) -> None:
    """Plot one metric as a single-series bar chart."""
    plot_df = prepare_plot_data(df, [metric])

    if plot_df.empty:
        raise ValueError(
            f"No data available to plot for metric '{metric}'."
        )

    x = range(len(plot_df))

    bars = ax.bar(
        x,
        plot_df[metric],
        color="#4C78A8",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        plot_df["combined_name"],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(metric)
    ax.set_title(metric)

    # Grid lines
    ax.grid(
        axis="y",
        alpha=0.25,
    )

    # Values above bars
    if show_values:
        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
        )

def plot_joint(
    df: pd.DataFrame,
    metrics: Sequence[str],
    ax: plt.Axes,
    *,
    show_values: bool = True,
) -> None:
    """
    Plot multiple metrics as grouped bars.

    Each combined_name is one group.
    Each metric gets one bar within the group.
    """
    plot_df = prepare_plot_data(df, metrics)

    if plot_df.empty:
        raise ValueError("No data available to plot.")

    n_groups = len(plot_df)
    n_metrics = len(metrics)

    x = list(range(n_groups))

    total_width = 0.8
    bar_width = total_width / n_metrics

    colors = get_bar_colors(metrics)

    for i, (metric, color) in enumerate(
        zip(metrics, colors)
    ):
        offsets = [
            position
            - total_width / 2
            + bar_width / 2
            + i * bar_width
            for position in x
        ]

        bars = ax.bar(
            offsets,
            plot_df[metric],
            width=bar_width,
            label=metric,
            color=color,
        )

        if show_values:
            ax.bar_label(
                bars,
                fmt="%.3f",
                padding=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df["combined_name"],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel("Metric value")
    ax.set_title("Model comparison")

    ax.legend()

    # Grid lines
    ax.grid(
        axis="y",
        alpha=0.25,
    )


# ---------------------------------------------------------------------------
# Main plotting API
# ---------------------------------------------------------------------------

def plot_metrics(
    df: pd.DataFrame,
    model_metadata: dict,
    *,
    models: str | Sequence[str] | None = None,
    sim_modes: str | Sequence[str] | None = None,
    split_modes: str | Sequence[str] | None = None,
    metrics: str | Sequence[str] | None = None,
    joint: bool = False,
    show_values: bool = True,
    output_dir: str | Path | None = None,
    show: bool = True,
    dpi: int = 300,
) -> list[Path]:
    """
    Create metric comparison plots.

    Parameters
    ----------
    df:
        Evaluation dataframe.

    model_metadata:
        Model metadata loaded from YAML.

    models:
        Models to include. None means all models.

    sim_modes:
        Simulation modes to include. None means all simulation modes.
        Models without simulation modes are always retained.

    split_modes:
        Split modes to include. None means all split modes.

    metrics:
        Metrics to plot. None is an error because plotting every metric
        by default is generally undesirable.

    joint:
        If False, create one figure per metric.
        If True, create one figure containing grouped bars for all metrics.

    show_values:
        If True, place metric values above bars for readability.

    output_dir:
        If supplied, save plots to this directory.

    show:
        Whether to display plots interactively.

    Returns
    -------
    list[Path]
        Paths to saved plots.
    """
    models = normalize_filter(models)
    sim_modes = normalize_filter(sim_modes)
    split_modes = normalize_filter(split_modes)
    metrics = normalize_filter(metrics)

    if metrics is None:
        raise ValueError(
            "At least one metric must be specified."
        )

    validate_metrics(df, metrics)

    filtered = filter_dataframe(
        df,
        model_metadata,
        models=models,
        sim_modes=sim_modes,
        split_modes=split_modes,
    )

    if filtered.empty:
        raise ValueError(
            "No rows matched the supplied filters."
        )

    # combined_name should uniquely identify a plotted model/configuration.
    if filtered["combined_name"].duplicated().any():
        duplicates = (
            filtered.loc[
                filtered["combined_name"].duplicated(keep=False),
                "combined_name",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "combined_name must uniquely identify a row after filtering. "
            f"Duplicates found: {duplicates}"
        )

    n_groups = len(filtered)

    output_paths: list[Path] = []

    if joint:
        width = calculate_figure_width(
            n_groups,
            joint=True,
        )

        fig, ax = plt.subplots(
            figsize=(width, 7),
            dpi=dpi,
        )

        plot_joint(
            filtered,
            metrics,
            ax,
            show_values=show_values
        )

        fig.tight_layout()

        filename = (
            make_plot_name(
                models,
                sim_modes,
                split_modes,
                metrics,
                joint=True,
            )
            + ".png"
        )

        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            path = output_path / filename
            fig.savefig(
                path,
                bbox_inches="tight",
            )

            output_paths.append(path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    else:
        for metric in metrics:
            width = calculate_figure_width(
                n_groups,
                joint=False,
            )

            fig, ax = plt.subplots(
                figsize=(width, 7),
                dpi=dpi,
            )

            plot_metric(
                filtered,
                metric,
                ax,
                show_values=show_values
            )

            fig.tight_layout()

            filename = (
                make_plot_name(
                    models,
                    sim_modes,
                    split_modes,
                    [metric],
                    joint=False,
                )
                + ".png"
            )

            if output_dir is not None:
                output_path = Path(output_dir)
                output_path.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                path = output_path / filename
                fig.savefig(
                    path,
                    bbox_inches="tight",
                )

                output_paths.append(path)

            if show:
                plt.show()
            else:
                plt.close(fig)

    return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_comma_sep_argument(
    value: str | None,
) -> list[str] | None:
    """Parse a comma-separated CLI argument."""
    if value is None:
        return None

    values = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    return values or None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot model evaluation metrics from an evaluation CSV."
        )
    )

    parser.add_argument(
        "--csv",
        default=DEFAULT_METRICS_FILE,
        help="Path to the evaluation dataframe CSV.",
    )

    parser.add_argument(
        "--models",
        help=(
            "Comma-separated models to include. "
            "Omit to include all models."
        ),
    )

    parser.add_argument(
        "--sim-modes",
        help=(
            "Comma-separated simulation modes to include. "
            "Omit to include all simulation modes."
        ),
    )

    parser.add_argument(
        "--split-modes",
        help=(
            "Comma-separated split modes to include. "
            "Omit to include all split modes."
        ),
    )

    parser.add_argument(
        "--metrics",
        help=(
            "Comma-separated metrics to plot."
        ),
    )

    parser.add_argument(
        "--joint",
        action="store_true",
        help=(
            "Plot all requested metrics together as grouped bars."
        ),
    )

    parser.add_argument(
        "--hide-values",
        action="store_true",
        default=False,
        help="Do not show metric values above each bar.",
    )

    parser.add_argument(
        "--model-metadata",
        default=DEFAULT_MODEL_METADATA,
        help="Path to the model metadata YAML file.",
    )

    parser.add_argument(
        "--output-dir",
        default=PLOTS_DIR,
        help="Directory in which to save plots.",
    )

    parser.add_argument(
        "--describe",
        action="store_true",
        help="Describe the dataframe and exit.",
    )

    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available metrics and exit.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display plots interactively.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for generated plots.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    df = load_dataframe(args.csv)

    if args.describe:
        describe_data(df)
        return

    if args.list_metrics:
        list_metrics(df)
        return

    if not args.metrics:
        parser.error(
            "--metrics is required unless --describe or "
            "--list-metrics is specified."
        )

    model_metadata = load_model_metadata(
        args.model_metadata
    )

    models = parse_comma_sep_argument(args.models)
    sim_modes = parse_comma_sep_argument(args.sim_modes)
    split_modes = parse_comma_sep_argument(args.split_modes)
    metrics = parse_comma_sep_argument(args.metrics)

    paths = plot_metrics(
        df,
        model_metadata,
        models=models,
        sim_modes=sim_modes,
        split_modes=split_modes,
        metrics=metrics,
        joint=args.joint,
        show_values=not args.hide_values,
        output_dir=args.output_dir,
        show=args.show,
        dpi=args.dpi,
    )

    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()