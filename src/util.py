
import os
import pandas as pd
import numpy as np
from itertools import product

PLAYS_PATH = os.path.join('data', 'pbp', 'pbp_2006_2025_plays.csv')
DRIVES_PATH = os.path.join('data', 'pbp', 'pbp_2006_2025_drives.csv')
METRICS_DIR = 'metrics'
ABSORBING_STATES = ["TD", "FG", "PUNT", "TURNOVER", "DOWNS", "HALF_END"]
ABSORBING_STATES_MAP = {
    241: "TD",
    242: "FG",
    243: "PUNT",
    244: "TURNOVER",
    245: "DOWNS",
    246: "HALF_END",
}

def load_data(path: str, plays_df=True) -> pd.DataFrame:
    """
    Load play-level data from a CSV file.
    """
    df = pd.read_csv(path)

    # Basic validation
    if plays_df:
        required_columns = {"cur_state_id", "next_state_id"}
        missing = required_columns - set(df.columns)
    else:
        required_columns = {"start_state_id", "drive_result_id"}
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

def build_state_to_field_pos(
    transition_matrix: pd.DataFrame,
    id_to_state: dict,
    field_pos_bins: dict,
    td_state: int,
) -> dict[int, float]:
    """
    Map state IDs to approximate field position using bin midpoints.

    Parameters
    ----------
    field_pos_bins:
        Mapping of field-position-bin ID -> (low, high).
    """

    field_pos_midpoints = {
        z: (low + high) / 2
        for z, (low, high) in field_pos_bins.items()
    }

    state_to_field_pos = {}
    all_states = set(transition_matrix.index) | set(transition_matrix.columns)

    for state_id in sorted(all_states):
        if state_id == td_state:
            state_to_field_pos[state_id] = 0.0
            break
        _, _, z = id_to_state[state_id]
        state_to_field_pos[state_id] = field_pos_midpoints[z]

    return state_to_field_pos

def drive_metrics_from_states(
    states: list[int],
    state_to_field_pos: dict[int, float],
    absorbing_outcomes: dict[int, str] = ABSORBING_STATES_MAP,
) -> dict:
    """
    Convert a simulated state sequence into drive-level characteristics.
    """

    if len(states) < 2:
        raise ValueError("Drive must contain at least two states.")

    final_state = states[-1]

    if final_state not in absorbing_outcomes:
        raise ValueError(
            f"Final state {final_state} is not a recognized absorbing state."
        )

    outcome = absorbing_outcomes[final_state]

    start_field_pos = state_to_field_pos[states[0]]
    try:
        end_field_pos = state_to_field_pos[states[-1]]
    except:
        # If end state is not TD, then there is no field position, so use field position of previous play state
        end_field_pos = state_to_field_pos[states[-2]]

    # Field position is measured as yards remaining to the opponent's
    # goal line, so moving toward the goal line decreases the value.
    net_field_position_change = start_field_pos - end_field_pos

    n_plays = len(states) - 1

    is_td = outcome == "TD"
    is_fg = outcome == "FG"
    is_score = is_td or is_fg

    points = (
        7 if is_td
        else 3 if is_fg
        else 0
    )

    return {
        "score": int(is_score),
        "td": int(is_td),
        "fg": int(is_fg),
        "points": points,
        "plays": n_plays,
        "yards_gained": net_field_position_change,
        "outcome": outcome,
        "final_state": final_state,
    }