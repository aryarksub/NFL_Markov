
import os
import pandas as pd
import numpy as np
from itertools import product

PBP_ALL_PATH = os.path.join('data', 'pbp', 'pbp_2006_2025_plays.csv')
METRICS_DIR = 'metrics'
ABSORBING_STATES = ["TD", "FG", "PUNT", "TURNOVER", "DOWNS", "HALF_END"]

def load_data(path: str) -> pd.DataFrame:
    """
    Load play-level data from a CSV file.
    """
    df = pd.read_csv(path)

    # Basic validation
    required_columns = {"cur_state_id", "next_state_id"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df

def get_ydstogo_cuts_labels():
    ydstogo_cuts = [0, 3, 7, 10, np.inf]
    ydstogo_labels = ["1-3", "4-7", "8-10", "11+"]
    return ydstogo_cuts, ydstogo_labels

def get_field_position_cuts_labels():
    # 5 yard bins on opponent half, 10 yard bins on own half
    field_position_cuts = [5*i for i in range(10)] + [10*i for i in range(5, 11)]
    field_position_labels =  [f'{5*i+1}-{5*(i+1)}' for i in range(10)] + [f'{10*i+1}-{10*(i+1)}' for i in range(5, 10)]
    return field_position_cuts, field_position_labels

def get_state_values(state_vars, value_lists):
    pairs = zip(state_vars, value_lists)
    return {
        var : list(range(1, len(val) + 1))
        for var, val in pairs
    }

def generate_state_ids(
    state_values,
    absorbing_states=None,
    start_id=1,
):
    """
    Generate IDs for the complete Cartesian product of state variables.

    Parameters
    ----------
    state_values : dict
        Dictionary mapping each state variable to all possible values.

        Example:
        {
            "down": [1, 2, 3, 4],
            "binned_ydstogo_id": [1, 2, 3, 4],
            "binned_field_position_id": list(range(1, 16)),
        }

    absorbing_states : list[str], optional
        Absorbing states. Their IDs are assigned immediately after all regular states.

    start_id : int, default=1
        ID assigned to the first regular state.

    Returns
    -------
    state_to_id : dict
        Maps state tuples -> integer IDs.

    id_to_state : dict
        Maps integer IDs -> state tuples / absorbing state names.

    absorbing_state_ids : dict
        Maps absorbing state names -> integer IDs.
    """

    if absorbing_states is None:
        absorbing_states = []

    # Preserve the order supplied by the user
    state_columns = list(state_values.keys())

    # Generate every possible combination
    all_states = list(
        product(
            *(state_values[column] for column in state_columns)
        )
    )

    # Regular state IDs
    state_to_id = {
        state: start_id + i
        for i, state in enumerate(all_states)
    }

    # Reverse lookup
    id_to_state = {
        state_id: state
        for state, state_id in state_to_id.items()
    }

    # Absorbing states follow regular states
    next_id = start_id + len(all_states)

    absorbing_state_ids = {
        state: next_id + i
        for i, state in enumerate(absorbing_states)
    }

    # Add absorbing states to reverse lookup
    id_to_state.update({
        state_id: state
        for state, state_id in absorbing_state_ids.items()
    })

    return (
        state_to_id,
        id_to_state,
        absorbing_state_ids,
    )