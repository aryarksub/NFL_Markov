import pandas as pd
import numpy as np
from sklearn.metrics import (
    log_loss, 
    brier_score_loss, 
    roc_auc_score, 
    average_precision_score, 
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from src import util

def predict_proba(
    transition_matrix: pd.DataFrame,
    current_states: pd.Series,
) -> pd.DataFrame:
    """
    Return P(next_state | current_state) for each observation.
    """

    known = current_states.isin(transition_matrix.index)
    return transition_matrix.loc[current_states[known]].reset_index(drop=True)

def predict(
    transition_matrix: pd.DataFrame,
    current_states: pd.Series,
) -> pd.Series:
    """
    Predict the most likely next state.
    """

    probs = predict_proba(
        transition_matrix,
        current_states
    )

    return probs.idxmax(axis=1)

def accuracy(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Calculate next-state classification accuracy.
    """

    return (y_true == y_pred).mean()

def top_k_accuracies(
    probabilities: pd.DataFrame,
    y_true: pd.Series,
    ks=(1, 3, 5, 10),
) -> dict:
    probs = probabilities.to_numpy()

    max_k = max(ks)

    top_indices = np.argpartition(
        probs,
        -max_k,
        axis=1,
    )[:, -max_k:]

    state_to_idx = {
        state: i
        for i, state in enumerate(probabilities.columns)
    }

    y_true_indices = y_true.map(state_to_idx).to_numpy()

    # Whether actual state is among top-k for each k
    results = {}

    for k in ks:
        correct = (
            top_indices[:, -k:] == y_true_indices[:, None]
        ).any(axis=1)

        results[f"top_{k}_accuracy"] = correct.mean()

    return results

def log_loss_score(
    y_true: pd.Series,
    probabilities: pd.DataFrame,
) -> float:
    """
    Calculate multiclass log loss.
    """

    return log_loss(
        y_true,
        probabilities,
        labels=probabilities.columns
    )

def first_down_metrics(
    transition_matrix: pd.DataFrame,
    test_df: pd.DataFrame,
    first_down_states=range(1, 61),
    cur_col="cur_state_id",
    next_col="next_state_id",
):
    """
    Calculate several metrics for predicting whether a play
    results in a first down.

    Returns:
        dict: Brier score, log loss, ROC AUC, and average precision.
    """

    # Probability of transitioning into a first-down state
    valid_first_down_states = [
        state
        for state in first_down_states
        if state in transition_matrix.columns
    ]

    first_down_probs = transition_matrix[
        valid_first_down_states
    ].sum(axis=1)

    # Map probability to each test play
    y_prob = test_df[cur_col].map(first_down_probs)

    # Actual first-down outcome
    y_true = test_df[next_col].isin(
        first_down_states
    ).astype(int)

    # Remove states unseen during training
    valid = y_prob.notna()

    y_true = y_true[valid]
    y_prob = y_prob[valid]

    return {
        "true_first_down_rate": y_true.mean(),
        "first_down_brier": brier_score_loss(
            y_true,
            y_prob,
        ),
        "first_down_log_loss": log_loss(
            y_true,
            y_prob,
        ),
        "first_down_roc_auc": roc_auc_score(
            y_true,
            y_prob,
        ),
        "first_down_average_precision": average_precision_score(
            y_true,
            y_prob,
        ),
    }

def evaluate_yards(
    transition_matrix: pd.DataFrame,
    test_df: pd.DataFrame,
    id_to_state: dict,
    field_pos_bins: dict,
    cur_col: str = "cur_state_id",
    yards_col: str = "yards_gained",
    next_col: str = "next_state_id"
) -> dict:
    """
    Evaluate Markov-chain predictions of yards gained.

    Two predictions are evaluated:

    1. Point prediction:
       Yards implied by the most likely next state.

    2. Expected prediction:
       Probability-weighted yards over all possible next states.

    Field position is measured relative to the opponent's end zone, so LOWER field-position values 
    mean closer to the opponent's end zone --> yards gained = current_field_pos - next_field_pos
    """

    # ---------------------------------------------------------
    # Build state_id -> field-position midpoint mapping
    # ---------------------------------------------------------

    state_to_field_pos = util.build_state_to_field_pos(
        transition_matrix=transition_matrix,
        id_to_state=id_to_state,
        field_pos_bins=field_pos_bins,
        td_state=241
    )

    field_pos = pd.Series(state_to_field_pos)

    # ---------------------------------------------------------
    # Keep test plays with states known to the model and 
    # exclude non-TD absorbing-state outcomes
    # ---------------------------------------------------------

    valid = (
        test_df[cur_col].isin(transition_matrix.index)
        & ~test_df[next_col].isin(range(242, 247))
    )
    eval_df = test_df.loc[valid].copy()

    # ---------------------------------------------------------
    # Current field position
    # ---------------------------------------------------------

    current_fp = eval_df[cur_col].map(field_pos)

    # ---------------------------------------------------------
    # 1. POINT YARDS
    # ---------------------------------------------------------

    # Most likely next state
    most_likely_next = transition_matrix.idxmax(axis=1)

    predicted_next_state = (
        eval_df[cur_col]
        .map(most_likely_next)
    )

    predicted_next_fp = predicted_next_state.map(field_pos)

    # Lower field position = more yards gained
    point_yards = current_fp - predicted_next_fp

    # ---------------------------------------------------------
    # 2. EXPECTED YARDS
    # ---------------------------------------------------------

    # Only states with a meaningful field position
    valid_current_states = [
        state
        for state in transition_matrix.index
        if state in field_pos.index
    ]

    # TD is valid; other absorbing states are not
    valid_next_states = [
        state
        for state in transition_matrix.columns
        if state in field_pos.index
    ]

    # Current field positions
    current_fp_array = field_pos.loc[
        valid_current_states
    ].to_numpy()[:, None]

    # Next field positions
    next_fp_array = field_pos.loc[
        valid_next_states
    ].to_numpy()[None, :]

    # Lower next FP means positive yards gained
    implied_yards = (
        current_fp_array - next_fp_array
    )

    # Restrict transition matrix to the same states
    valid_transition_matrix = transition_matrix.loc[
        valid_current_states,
        valid_next_states,
    ]

    expected_yards_by_state = (
        valid_transition_matrix.to_numpy()
        * implied_yards
    ).sum(axis=1)

    expected_yards_by_state = pd.Series(
        expected_yards_by_state,
        index=valid_current_states,
    )

    expected_yards = eval_df[cur_col].map(
        expected_yards_by_state
    )

    # ---------------------------------------------------------
    # Actual yards
    # ---------------------------------------------------------

    actual_yards = eval_df[yards_col]

    valid_rows = (
        actual_yards.notna()
        & point_yards.notna()
        & expected_yards.notna()
    )

    actual_yards = actual_yards[valid_rows]
    point_yards = point_yards[valid_rows]
    expected_yards = expected_yards[valid_rows]

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    def calculate_metrics(y_true, y_pred, prefix):
        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "correlation": y_true.corr(y_pred),
            "r2": r2_score(y_true, y_pred),
        }

        return {
            f"{prefix}_{metric}": value
            for metric, value in metrics.items()
        }

    return {
        **calculate_metrics(
            actual_yards,
            point_yards,
            "point_yards",
        ),
        **calculate_metrics(
            actual_yards,
            expected_yards,
            "expected_yards",
        ),
        "yards_plays_evaluated": len(actual_yards),
    }

def binary_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    prefix: str,
) -> dict:
    """
    Compute classification metrics for a deterministic binary prediction.
    """

    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        f"{prefix}_recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        f"{prefix}_f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        f"{prefix}_actual_rate": y_true.mean(),
        f"{prefix}_predicted_rate": y_pred.mean(),
    }

def continuous_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    prefix: str,
) -> dict:
    return {
        f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}_rmse": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
    }

def multiclass_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    prefix: str = "final_state",
) -> dict:
    """
    Compute multiclass classification metrics.
    """

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_macro_precision": precision_macro,
        f"{prefix}_macro_recall": recall_macro,
        f"{prefix}_macro_f1": f1_macro,
        f"{prefix}_weighted_precision": precision_weighted,
        f"{prefix}_weighted_recall": recall_weighted,
        f"{prefix}_weighted_f1": f1_weighted,
    }