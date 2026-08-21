import pandas as pd
import numpy as np
import os
import argparse
import json
import re

from src.markov import evals
from src import util

MODEL_NAME = 'simple_markov'
SIMPLE_MARKOV_METRICS_DIR = os.path.join(util.METRICS_DIR, MODEL_NAME)
os.makedirs(SIMPLE_MARKOV_METRICS_DIR, exist_ok=True)

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

def simulate_drive_greedy(
    start_state: int,
    transition_matrix: pd.DataFrame,
    absorbing_states: set = set(util.ABSORBING_STATES_MAP.keys()),
    max_plays: int = 50,
    **kwargs,
) -> list[int]:
    """
    Simulate a drive autoregressively using the maximum-probability (greedy) transition at each state.

    Returns
    -------
    list[int]
        Sequence of states, including the starting state and the final absorbing state.
    """
    states = [start_state]
    current_state = start_state

    for _ in range(max_plays):
        # Stop once we reach an absorbing state.
        if current_state in absorbing_states:
            return states

        if current_state not in transition_matrix.index:
            raise ValueError(
                f"State {current_state} is not present in transition matrix."
            )

        probabilities = transition_matrix.loc[current_state]

        # Ignore zero-probability / missing transitions.
        probabilities = probabilities.dropna()
        probabilities = probabilities[probabilities > 0]

        if probabilities.empty:
            raise ValueError(
                f"State {current_state} has no positive-probability transitions."
            )

        next_state = int(probabilities.idxmax())

        states.append(next_state)
        current_state = next_state

    raise RuntimeError(
        f"Drive did not reach an absorbing state within {max_plays} plays. "
        f"Last state: {current_state}"
    )

def simulate_drive_sample(
    start_state: int,
    transition_matrix: pd.DataFrame,
    absorbing_states: set = set(util.ABSORBING_STATES_MAP.keys()),
    max_plays: int = 50,
    **kwargs,
) -> list[int]:
    """
    Simulate a drive by sampling each transition according to its
    probability in the transition matrix.

    Each call produces one stochastic realization of the drive. Unlike
    greedy or maximum-probability simulation, the same starting state
    may produce different drives across calls.

    Parameters
    ----------
    start_state
        Initial state of the drive.

    transition_matrix
        Transition probability matrix. Rows correspond to current states
        and columns correspond to possible next states.

    absorbing_states
        States at which the drive terminates.

    max_plays
        Maximum number of transitions to simulate.

    **kwargs
        Additional arguments accepted for compatibility with other
        simulation methods. Not used.

    Returns
    -------
    list[int]
        Sequence of states, including the starting state and final
        absorbing state.
    """
    states = [start_state]
    current_state = start_state

    for _ in range(max_plays):
        # Stop once we reach an absorbing state.
        if current_state in absorbing_states:
            return states

        if current_state not in transition_matrix.index:
            raise ValueError(
                f"State {current_state} is not present in transition matrix."
            )

        probabilities = transition_matrix.loc[current_state]

        # Ignore zero-probability / missing transitions.
        probabilities = probabilities.dropna()
        probabilities = probabilities[probabilities > 0]

        if probabilities.empty:
            raise ValueError(
                f"State {current_state} has no positive-probability transitions."
            )

        # Normalize in case the row does not sum to exactly 1 due to
        # floating-point precision or preprocessing.
        probabilities = probabilities / probabilities.sum()

        next_state = int(
            np.random.choice(
                probabilities.index,
                p=probabilities.values,
            )
        )

        states.append(next_state)
        current_state = next_state

    raise RuntimeError(
        f"Drive did not reach an absorbing state within {max_plays} plays. "
        f"Last state: {current_state}"
    )

def simulate_drive_max_prob(
    start_state: int,
    transition_matrix: pd.DataFrame,
    absorbing_states: set = set(util.ABSORBING_STATES_MAP.keys()),
    max_plays: int = 50,
    **kwargs,
) -> list[int]:
    """
    Reconstruct the exact maximum-probability drive for a starting state.

    `V` and `best_next` must be supplied through `kwargs` and should be
    precomputed once using `build_max_probability_policy()`.

    `transition_matrix` is retained for interface compatibility with
    other drive simulation functions but is not used directly.
    """
    V = kwargs["V"]
    best_next = kwargs["best_next"]

    start_key = (start_state, max_plays)

    if start_key not in V:
        raise ValueError(
            f"State {start_state} is not present in the precomputed policy."
        )

    if V[start_key] <= 0:
        raise RuntimeError(
            f"No absorbing state is reachable from state {start_state} "
            f"within {max_plays} plays."
        )

    states = [start_state]
    current_state = start_state
    remaining_plays = max_plays

    while current_state not in absorbing_states:
        key = (current_state, remaining_plays)

        if key not in best_next:
            raise RuntimeError(
                f"No policy entry exists for state {current_state} "
                f"with {remaining_plays} plays remaining."
            )

        next_state = best_next[key]

        if next_state is None:
            raise RuntimeError(
                f"Could not reconstruct a path from state {current_state}."
            )

        states.append(next_state)
        current_state = next_state
        remaining_plays -= 1

    return states

def compute_drive_level_metrics(
    true_drives_df: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    state_to_field_pos: dict[int, float],
    sim_mode: str = 'greedy',
):
    test_drives_df = true_drives_df.copy()
    drive_metrics = dict()

    sim_drives = simulate_drives(
        drive_df=test_drives_df,
        transition_matrix=transition_matrix,
        state_to_field_pos=state_to_field_pos,
        mode=sim_mode
    )

    comparison = pd.DataFrame(index=test_drives_df.index)
    # Map of column names in simulated drive dataframe -> ground truth drive-level dataframe
    cols_mapping = {
        "score": "scored",
        "td": "td_in_drive",
        "fg": "fg_in_drive",
        "points": "points",
        "plays": "num_plays",
        "yards_gained": "net_field_position_change",
        "final_state": "drive_result_id",
    }

    for metric, actual_col in cols_mapping.items():
        comparison[f"actual_{metric}"] = test_drives_df[actual_col].values
        comparison[f"pred_{metric}"] = sim_drives[metric].values

    using_averaged_metrics = 'sample' in sim_mode and sim_mode != 'sample1'

    # ---------------------------------------------------------
    # Binary outcomes
    # ---------------------------------------------------------

    for metric in ["score", "td", "fg"]:
        if using_averaged_metrics:
            drive_metrics.update(
                evals.continuous_metrics(
                    comparison[f"actual_{metric}"],
                    comparison[f"pred_{metric}"],
                    prefix=f'drive_{metric}',
                    include_brier=True # do Brier since target is binary
                )
            )
        else:    
            # Cannot use binary measurement on averaged values (they are continuous, not binary)
            drive_metrics.update(
                evals.binary_metrics(
                    comparison[f"actual_{metric}"],
                    comparison[f"pred_{metric}"],
                    prefix=f'drive_{metric}',
                )
            )

    # ---------------------------------------------------------
    # Continuous / count outcomes
    # ---------------------------------------------------------

    for metric in ["points", "plays", "yards_gained"]:
        drive_metrics.update(
            evals.continuous_metrics(
                comparison[f"actual_{metric}"],
                comparison[f"pred_{metric}"],
                prefix=f'drive_{metric}',
                include_brier=False # don't do Brier since target is not binary
            )
        )

    # ---------------------------------------------------------
    # Multiclass final-state prediction
    # ---------------------------------------------------------

    for metric in ['final_state']:
        drive_metrics.update(
            evals.multiclass_metrics(
                comparison[f"actual_{metric}"],
                comparison[f"pred_{metric}"],
                prefix=f'drive_{metric}',
            )
        )

    # Useful overall counts
    drive_metrics["drives_evaluated"] = len(comparison)

    return drive_metrics, comparison

def simulate_drives(
    drive_df: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    state_to_field_pos: dict[int, float],
    absorbing_states: set[int] = set(util.ABSORBING_STATES_MAP.keys()),
    absorbing_outcomes: dict[int, str] = util.ABSORBING_STATES_MAP,
    start_state_col: str = "start_state_id",
    max_plays: int = 100,
    mode: str = 'greedy',
) -> pd.DataFrame:
    """
    Simulate one deterministic Markov-chain drive for every drive
    in drive_df.
    """

    sim_kwargs = {}
    num_sims = 1

    if mode == 'greedy':
        sim_func = simulate_drive_greedy
    elif 'sample' in mode:
        sim_func = simulate_drive_sample
        num_sims = int(mode[len('sample'):].strip())
    elif mode == 'max_prob':
        sim_func = simulate_drive_max_prob
        V, best_next = util.build_max_probability_policy(
            transition_matrix=transition_matrix,
            absorbing_states=absorbing_states,
            max_plays=max_plays,
        )
        sim_kwargs = {
            "V": V,
            "best_next": best_next,
        }
    else:
        raise ValueError(f"Unsupported drive simulation method: {mode}")

    results = []
    proc_rows = 0

    print(f'Processed 0/{len(drive_df)} drives')
    for _, row in drive_df.iterrows():
        start_state = row[start_state_col]

        sample_metrics = []

        for i in range(1, num_sims+1):
            states = sim_func(
                start_state=start_state,
                transition_matrix=transition_matrix,
                absorbing_states=absorbing_states,
                max_plays=max_plays,
                **sim_kwargs
            )

            metrics = util.drive_metrics_from_states(
                states=states,
                state_to_field_pos=state_to_field_pos,
                absorbing_outcomes=absorbing_outcomes,
            )

            sample_metrics.append(metrics)

        # Average numeric metrics across stochastic samples.
        metrics = {}
        for key in sample_metrics[0]:
            values = [sample[key] for sample in sample_metrics]

            if key in {"final_state"}:
                modes = pd.Series(values).mode().tolist()
                metrics[key] = np.random.choice(modes)
            else:
                metrics[key] = np.mean(values)

        metrics["start_state"] = start_state
        # state_sequence is not meaningful when sampling, so it's ok to just set this to the last set of simulated states
        metrics["state_sequence"] = states

        results.append(metrics)
        proc_rows += 1
        if proc_rows % 5000 == 0 or proc_rows == len(drive_df):
            print(f'Processed {proc_rows}/{len(drive_df)} drives')

    return pd.DataFrame(results)

def evaluate_markov_chain(
    transition_matrix: pd.DataFrame,
    test_plays_df: pd.DataFrame,
    test_drives_df: pd.DataFrame,
    cur_col: str = "cur_state_id",
    next_col: str = "next_state_id",
    sim_mode: str = "sample1",
    save_sims: bool = False,
) -> dict:
    """
    Evaluate a Markov chain on held-out data.
    """
    print(f'Evaluating Markov chain with drive simulation mode {sim_mode}')

    known_current = test_plays_df[cur_col].isin(
        transition_matrix.index
    )

    known_next = test_plays_df[next_col].isin(
        transition_matrix.columns
    )

    # Only evaluate transitions where both states were observed during training.
    known = known_current & known_next

    eval_df = test_plays_df.loc[known].reset_index(drop=True)

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

    state_to_field_pos = util.build_state_to_field_pos(
        transition_matrix=transition_matrix,
        id_to_state=id_to_state,
        field_pos_bins=field_position_dict,
        td_state=241
    )

    ##############################################################
    # Play-level metrics
    ##############################################################
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

    ##############################################################
    # Drive-level metrics
    ##############################################################

    drive_metrics, sim_comp_df = compute_drive_level_metrics(
        true_drives_df=test_drives_df,
        transition_matrix=transition_matrix,
        state_to_field_pos=state_to_field_pos,
        sim_mode=sim_mode
    )
    metrics |= drive_metrics

    if save_sims:
        sim_comp_df.to_csv(
            os.path.join(SIMPLE_MARKOV_METRICS_DIR, f'{MODEL_NAME}_{sim_mode}_sims.csv'),
            index=False
        )

    return {
        "model" : MODEL_NAME,
        "sim_mode" : sim_mode,
        "metrics" : metrics
    }


def driver(
    plays_path=util.PLAYS_PATH,
    drives_path=util.DRIVES_PATH,
    train_test_split_modes=['all', 'yearly', 'bundle', 'cut2020'],
    train_frac=0.75,
    sim_mode='sample1',
    save_sims=False,
):
    print('Loading plays data from', plays_path)
    plays_df = util.load_data(plays_path)

    print('Loading drives data from', drives_path)
    drives_df = util.load_data(drives_path, plays_df=False)

    for train_test_split_mode in train_test_split_modes:
        metrics_dir = os.path.join(SIMPLE_MARKOV_METRICS_DIR, train_test_split_mode)
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_save_path = os.path.join(
            metrics_dir, 
            f'{MODEL_NAME}_{sim_mode}_{train_test_split_mode}.json'
        )

        train_plays_dfs, test_plays_dfs, test_drives_dfs = util.get_train_test_sets(
            plays_df=plays_df,
            drives_df=drives_df,
            split_mode=train_test_split_mode,
            train_frac=train_frac
        )

        eval_data_list = []
        weights = []

        for train_plays_df, test_plays_df, test_drives_df in zip(
            train_plays_dfs,
            test_plays_dfs,
            test_drives_dfs,
        ):
            print(f"Training on {len(train_plays_df)} plays, evaluating on {len(test_plays_df)} plays and {len(test_drives_df)} drives.")

            transition_matrix = fit_markov_chain(train_plays_df)

            eval_data = evaluate_markov_chain(
                transition_matrix=transition_matrix,
                test_plays_df=test_plays_df,
                test_drives_df=test_drives_df,
                sim_mode=sim_mode,
                save_sims=save_sims,
            )
            eval_data["split_mode"] = train_test_split_mode
            eval_data["combined_name"] = f"{MODEL_NAME}_{sim_mode}_{train_test_split_mode}"

            eval_data_list.append(eval_data)
            weights.append(len(test_plays_df))
            
        eval_data = util.weighted_average_eval_data(
            eval_data_list,
            weights,
        )

        with open(metrics_save_path, "w") as file:
            json.dump(util.round_floats(eval_data, 3), file, indent=4)

def parse_sim_mode(value: str) -> str:
    if value in {"greedy", "max_prob"}:
        return value

    if re.fullmatch(r"sample\d+", value):
        return value

    raise argparse.ArgumentTypeError(
        "sim-mode must be 'greedy', 'max_prob', or 'sample<number>' (e.g. sample10, sample20)"
    )

def parse_train_test_split_modes(value: str) -> list[str]:
    modes = [mode.strip() for mode in value.split(",")]

    if not modes or any(not mode for mode in modes):
        raise argparse.ArgumentTypeError(
            "train-test-split-mode must contain one or more modes"
        )

    for mode in modes:
        if mode in {"all", "yearly", "bundle"}:
            continue

        if re.fullmatch(r"cut\d+", mode):
            continue

        raise argparse.ArgumentTypeError(
            f"invalid train-test-split-mode '{mode}': "
            "must be 'all', 'yearly', 'bundle', or 'cut<number>' (e.g. cut2020)"
        )

    return modes

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Markov chain on PBP data."
    )

    parser.add_argument(
        "--plays-path", "--plays",
        type=str,
        default=util.PLAYS_PATH,
        help="Path to the plays data file.",
    )

    parser.add_argument(
        "--drives-path", "--drives",
        type=str,
        default=util.DRIVES_PATH,
        help="Path to the drives data file.",
    )
    
    parser.add_argument(
        "--train-test-split-modes", "--split-modes", "-ttsm",
        type=parse_train_test_split_modes,
        default=["all", "yearly", "bundle", "cut2020"],
        help="Train/Test split modes, comma-separated: all, yearly, bundle, or cut<YEAR>",
    )

    parser.add_argument(
        "--train-frac", "--tfrac",
        type=float,
        default=0.75,
        help="Fraction of games to use in training dataset",
    )

    parser.add_argument(
        "--sim-mode", "--mode", "-sm",
        type=parse_sim_mode,
        default="sample1",
        help="Drive simulation mode: greedy, max_prob, or sample<N>",
    )

    parser.add_argument(
        "--save-sims", "--save", "-ss",
        action="store_true",
        default=False,
        help="Save drive simulation dataframe",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    driver(
        plays_path=args.plays_path,
        drives_path=args.drives_path,
        train_test_split_modes=args.train_test_split_modes,
        train_frac=args.train_frac,
        sim_mode=args.sim_mode,
        save_sims=args.save_sims,
    )