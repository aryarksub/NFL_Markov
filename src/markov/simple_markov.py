import pandas as pd
import numpy as np
import os
import argparse
import json

from src.markov import evals
from src import util

def fit_markov_chain(
    df: pd.DataFrame,
    cur_col: str = "cur_state_id",
    next_col: str = "next_state_id",
) -> pd.DataFrame:
    """
    Estimate a first-order Markov transition matrix.

    Returns
    -------
    pd.DataFrame
        Rows are current states, columns are next states.
        Each row contains transition probabilities.
    """
    print('Fitting Markov chain')

    # All states appearing anywhere in the data
    states = np.sort(
        pd.unique(
            pd.concat([
                df[cur_col],
                df[next_col]
            ])
        )
    )

    # Count transitions
    counts = pd.crosstab(
        df[cur_col],
        df[next_col]
    )

    # Make matrix square
    counts = counts.reindex(
        index=states,
        columns=states,
        fill_value=0
    )

    # Normalize rows
    transition_matrix = counts.div(
        counts.sum(axis=1),
        axis=0
    ).fillna(0)

    return transition_matrix

def evaluate_markov_chain(
    transition_matrix: pd.DataFrame,
    test_df: pd.DataFrame,
    cur_col: str = "cur_state_id",
    next_col: str = "next_state_id",
) -> dict:
    """
    Evaluate a Markov chain on held-out data.
    """
    print('Evaluating Markov chain')

    known_current = test_df[cur_col].isin(
        transition_matrix.index
    )

    known_next = test_df[next_col].isin(
        transition_matrix.columns
    )

    # Only evaluate transitions where both states were observed during training.
    known = known_current & known_next

    eval_df = test_df.loc[known].reset_index(drop=True)

    probabilities = transition_matrix.loc[
        eval_df[cur_col]
    ].reset_index(drop=True)

    y_true = eval_df[next_col]

    _, ydstogo_labels = util.get_ydstogo_cuts_labels()
    field_position_cuts, field_position_labels = util.get_field_position_cuts_labels()
    state_values = util.get_state_values(
        ["down", "binned_ydstogo_id", "binned_field_position_id"],
        [
            [1,2,3,4],
            ydstogo_labels,
            field_position_labels
        ]
    )
    _, id_to_state, _ = util.generate_state_ids(
        state_values=state_values,
        absorbing_states=util.ABSORBING_STATES,
    )
    # Indices used in state tuple start at 1
    field_position_dict = {
        i+1 : (field_position_cuts[i], field_position_cuts[i+1]) for i in range(len(field_position_cuts) - 1)
    }

    accuracies = evals.top_k_accuracies(
        probabilities, y_true, [1,3,5,10]
    )

    first_down_metrics = evals.first_down_metrics(
        transition_matrix=transition_matrix,
        test_df=eval_df
    )

    yards_gained_metrics = evals.evaluate_yards(
        transition_matrix=transition_matrix,
        test_df=eval_df,
        id_to_state=id_to_state,
        field_pos_bins=field_position_dict
    )

    other_metrics = {
        "state_log_loss": evals.log_loss_score(
            y_true,
            probabilities,
        ),
        "state_plays_evaluated": len(eval_df),
        "state_plays_excluded": (~known).sum().item(),
    }
    metrics = accuracies | first_down_metrics | yards_gained_metrics | other_metrics

    return {
        "model" : "simple_markov",
        "metrics" : metrics
    }

def driver(
    pbp_path=util.PBP_ALL_PATH,
    yr_cutpoint=2020,
    metrics_save_path=os.path.join('metrics', 'simple_markov.json')
):
    print('Loading data from', pbp_path)
    df = util.load_data(pbp_path)

    print('Splitting data at', yr_cutpoint)
    train_df = df[df["season"] <= yr_cutpoint]
    test_df = df[df["season"] > yr_cutpoint]

    transition_matrix = fit_markov_chain(train_df)

    eval_data = evaluate_markov_chain(
        transition_matrix,
        test_df
    )

    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w") as file:
        json.dump(eval_data, file, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Markov chain on PBP data."
    )

    parser.add_argument(
        "--pbp-path",
        type=str,
        default=util.PBP_ALL_PATH,
        help="Path to the PBP data file.",
    )

    parser.add_argument(
        "--yr-cutpoint", '--yr-cut',
        type=int,
        default=2020,
        help="Last season included in training data. Seasons after this are used for testing.",
    )

    parser.add_argument(
        "--metrics-save-path", "--metrics-path", "-msp",
        type=str,
        default=os.path.join('metrics', 'simple_markov.json'),
        help="Path to save model metrics.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    driver(
        pbp_path=args.pbp_path,
        yr_cutpoint=args.yr_cutpoint,
        metrics_save_path=args.metrics_save_path
    )