import os
import pandas as pd
import numpy as np
import argparse

import src.util as util

PBP_DIR = os.path.join('data', 'pbp')

def bin_column(series, cutpoints, labels):
    binned = pd.cut(
        series,
        bins=cutpoints,
        labels=labels,
        include_lowest=True,
        right=True
    )

    bin_ids = (
        pd.cut(
            series,
            bins=cutpoints,
            labels=False,
            include_lowest=True,
            right=True
        )
        + 1
    )

    return binned, bin_ids

def prepare_play_level_data(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare nflfastR play-by-play data for the NFL Markov chain model.
    States used for the Markov chain incorporate down, yards to go, and field position. We
    use six absorbing states: TD, FG, PUNT, HALF_END, TURNOVER, DOWNS.
    Only plays occurring in quarters 1-4 (no OT) and on downs 1-4 (no kickoffs, extra points, etc.) are retained.
    Only regular season games are retained.
    """

    df = pbp.copy()

    # =========================================================
    # 1. Basic retention policy 
    # - Downs 1-4
    # - Quarters 1-4
    # - Regular season
    # =========================================================

    df = df[
        df["down"].isin([1, 2, 3, 4])
        & df["qtr"].isin([1, 2, 3, 4])
        & (df["season_type"] == 'REG')
    ].copy()

    # Sort chronologically within games/drives
    df = df.sort_values(
        [
            "game_date",
            "game_id",
            "drive",
            "game_seconds_remaining",
            "play_id",
            "down"
        ],
        ascending=[
            True,   # game_date
            True,   # game_id
            True,   # drive
            False,  # game_seconds_remaining
            True,   # play_id
            True    # down
        ]
    ).reset_index(drop=True)

    # =========================================================
    # 2. Identifiers
    # =========================================================

    # Rename drive to make its purpose explicit
    df["drive_number"] = df["drive"]

    # Play number within drive
    df["play_number"] = (
        df.groupby(["game_id", "drive_number"])
          .cumcount()
          .add(1)
    )

    # =========================================================
    # 3. Field position
    # =========================================================

    # yardline_100 represents yards from opponent end zone
    # Example:
    #   75 -> own 25
    #   50 -> midfield
    #   25 -> opponent 25
    df["field_position"] = df["yardline_100"]

    # =========================================================
    # 4. Bin yards to go
    # =========================================================

    ydstogo_cuts, ydstogo_labels = util.get_ydstogo_cuts_labels()
    df["binned_ydstogo"], df["binned_ydstogo_id"] = bin_column(
        df["ydstogo"],
        cutpoints=ydstogo_cuts,
        labels=ydstogo_labels
    )

    # =========================================================
    # 5. Bin field position
    # =========================================================

    field_position_cuts, field_position_labels = util.get_field_position_cuts_labels()
    df["binned_field_position"], df["binned_field_position_id"] = bin_column(
        df["yardline_100"],
        cutpoints=field_position_cuts,
        labels=field_position_labels
    )

    # =========================================================
    # 6. Current Markov state
    # =========================================================

    state_values = util.get_state_values(
        ["down", "binned_ydstogo_id", "binned_field_position_id"],
        [
            [1,2,3,4],
            ydstogo_labels,
            field_position_labels
        ]
    )
    state_to_id, id_to_state, absorbing_state_ids = util.generate_state_ids(
        state_values=state_values,
        absorbing_states=util.ABSORBING_STATES,
    )
    state_columns = list(state_values.keys())

    valid_state = df[state_columns].notna().all(axis=1)

    df["cur_state"] = None
    df.loc[valid_state, "cur_state"] = pd.Series(
        [
            tuple(row)
            for row in df.loc[valid_state, state_columns].astype(int).to_numpy()
        ],
        index=df.index[valid_state],
        dtype="object",
    )
    df["cur_state_id"] = df["cur_state"].map(state_to_id)

    # =========================================================
    # 7. Absorbing states
    # =========================================================

    df["absorbing_state"] = None

    # ---------------------------------------------------------
    # TD
    # ---------------------------------------------------------
    df.loc[
        df["td_team"].notna()
        & df["td_team"].eq(df["posteam"]),
        "absorbing_state"
    ] = "TD"

    # ---------------------------------------------------------
    # FG made
    # ---------------------------------------------------------
    df.loc[
        df["absorbing_state"].isna()
        & df["field_goal_result"].eq("made"),
        "absorbing_state"
    ] = "FG"

    # ---------------------------------------------------------
    # Punt
    # ---------------------------------------------------------
    df.loc[
        df["absorbing_state"].isna()
        & df["punt_attempt"].eq(1),
        "absorbing_state"
    ] = "PUNT"

    # ---------------------------------------------------------
    # Turnover
    #
    # interception
    # fumble lost
    # safety
    # blocked FG
    # ---------------------------------------------------------
    df.loc[
        df["absorbing_state"].isna()
        & (
            df["interception"].eq(1)
            | df["fumble_lost"].eq(1)
            | df["safety"].eq(1)
            | df["field_goal_result"].eq("blocked")
        ),
        "absorbing_state"
    ] = "TURNOVER"

    # ---------------------------------------------------------
    # Downs
    #
    # fourth-down failure
    # missed FG
    # ---------------------------------------------------------
    df.loc[
        df["absorbing_state"].isna()
        & (
            df["fourth_down_failed"].eq(1)
            | df["field_goal_result"].eq("missed")
        ),
        "absorbing_state"
    ] = "DOWNS"

    # ---------------------------------------------------------
    # Half/game end
    #
    # qtr = 2 or 4 AND quarter_end = true
    # ---------------------------------------------------------
    df.loc[
        df["absorbing_state"].isna()
        & df["qtr"].isin([2, 4])
        & df["quarter_end"].eq(1),
        "absorbing_state"
    ] = "HALF_END"

    # Additional processing for HALF_END state: Identify drives where every play has no absorbing state
    drive_has_absorbing = (
        df.groupby(["game_id", "drive_number"])["absorbing_state"]
        .transform("count")
    )

    # Last row of each drive
    last_play = (
        df.groupby(["game_id", "drive_number"])
        .tail(1)
        .index
    )

    # Drives whose last play is in Q2 or Q4 and have no absorbing state
    half_end = (
        df.index.isin(last_play)
        & df["qtr"].isin([2, 4])
        & (drive_has_absorbing == 0)
    )

    df.loc[half_end, "absorbing_state"] = "HALF_END"

    # Remove all remaining drives that have no absorbing state
    drive_has_absorbing = (
        df.groupby(["game_id", "drive_number"])["absorbing_state"]
        .transform("count")
    )

    df = df[drive_has_absorbing > 0].copy()

    df["absorbing_state_id"] = (
        df["absorbing_state"].map(absorbing_state_ids)
    )

    # =========================================================
    # 8. Next state
    # =========================================================

    df["next_state"] = (
        df.groupby(
            ["game_id", "drive_number"],
            sort=False
        )["cur_state"]
        .shift(-1)
    )

    # For terminal plays, the next state is the absorbing state
    terminal = df["absorbing_state"].notna()

    df.loc[terminal, "next_state"] = (
        df.loc[terminal, "absorbing_state"]
    )

    def get_next_state_id(state):
        if pd.isna(state):
            return np.nan

        if isinstance(state, tuple):
            return state_to_id.get(state, np.nan)

        if isinstance(state, str):
            return absorbing_state_ids.get(state, np.nan)

        return np.nan

    df["next_state_id"] = df["next_state"].apply(get_next_state_id)

    # =========================================================
    # 9. Remove problematic drives
    # =========================================================

    # Drop drives where there is a row (play) with next_state = None
    invalid_drives = (
        df.loc[df["next_state"].isna(), ["game_id", "drive_number"]]
        .drop_duplicates()
    )

    df = df[
        ~df.set_index(["game_id", "drive_number"]).index.isin(
            invalid_drives.set_index(["game_id", "drive_number"]).index
        )
    ].reset_index(drop=True)

    # Drop drives where there are two teams with possession (2 posteam values)
    posteam_count = (
        df.groupby(["game_id", "drive_number"])["posteam"]
        .transform("nunique")
    )

    df = df[posteam_count != 2].copy()

    # Manual drive removal
    drives_to_remove = [
        # Remove drives with problematic information related to blocked FGs
        ("2013_05_SD_OAK", 12),
        # Remove drives where there is a play transition "backwards" (4th -> 3rd or 3rd -> 2nd down)
        ("2006_01_PHI_HOU", 7),
        ("2006_13_MIN_CHI", 11),
        ("2007_02_HOU_CAR", 10),
        ("2008_02_NE_NYJ", 1),
        ("2008_04_ATL_CAR", 9),
        ("2010_09_NYG_SEA", 19),
        ("2011_10_DET_CHI", 14),
        ("2011_15_NYJ_PHI", 15),
        ("2013_06_GB_BAL", 11),
        ("2018_10_JAX_IND", 9),
        ("2020_01_LV_CAR", 13),
        ("2021_18_NE_MIA", 14),
    ]

    remove_index = pd.MultiIndex.from_tuples(
        drives_to_remove,
        names=["game_id", "drive_number"],
    )

    df_index = pd.MultiIndex.from_frame(
        df[["game_id", "drive_number"]]
    )

    df = df.loc[~df_index.isin(remove_index)].copy()

    ### Drop all drives where the first play is problematic
    first_play = df["play_number"].eq(1)

    # Conditions that make a drive problematic
    bad_first_play = (
        # Drive doesn't start at first down
        (df["down"] > 1)
        | (
            (df["down"] == 1)
            & (
                # Drive starts at 1st down and more than 10
                (df["ydstogo"] > 10)
                | (
                    # Drive starts at 1st down and less than 10, but not goal-to-go situation
                    (df["ydstogo"] < 10)
                    & (df["ydstogo"] != df["field_position"])
                )
            )
        )
    )

    # Get the game/drive combinations with a problematic first play
    bad_drives = df.loc[
        first_play & bad_first_play,
        ["game_id", "drive_number"]
    ].drop_duplicates()

    # Drop all plays belonging to those drives
    df = df.merge(
        bad_drives.assign(_bad_drive=True),
        on=["game_id", "drive_number"],
        how="left"
    )

    df = df[df["_bad_drive"].ne(True)].drop(columns="_bad_drive").copy()

    # =========================================================
    # 10. Drive result
    # =========================================================

    # The drive result is the absorbing state of the drive's terminal play. Propagate that result to every 
    # play in the drive.
    drive_result = (
        df.groupby(["game_id", "drive_number"])["absorbing_state"]
          .transform("last")
    )
    drive_result_id = (
        df.groupby(["game_id", "drive_number"])["absorbing_state_id"]
          .transform("last")
    )

    df["drive_result"] = drive_result
    df["drive_result_id"] = drive_result_id
    df["drive_success"] = df["drive_result"].isin(["TD", "FG"])

    # =========================================================
    # 11. Final columns
    # =========================================================

    cols_to_keep = [
        # -----------------------------------------------------
        # Game identifiers
        # -----------------------------------------------------
        "season",
        "season_type",
        "week",
        "game_id",
        "game_date",

        # -----------------------------------------------------
        # Drive/play identifiers
        # -----------------------------------------------------
        "drive_number",
        "play_id",
        "play_number",

        # -----------------------------------------------------
        # Teams
        # -----------------------------------------------------
        "posteam",
        "posteam_type",
        "defteam",

        # -----------------------------------------------------
        # Play type
        # -----------------------------------------------------
        "play_type",
        "pass_attempt",
        "rush_attempt",

        # -----------------------------------------------------
        # Current state
        # -----------------------------------------------------
        "down",
        "ydstogo",
        "yardline_100",
        "field_position",
        "binned_ydstogo",
        "binned_ydstogo_id",
        "binned_field_position",
        "binned_field_position_id",
        "cur_state",
        "cur_state_id",

        # -----------------------------------------------------
        # Transition
        # -----------------------------------------------------
        "next_state",
        "next_state_id",

        # -----------------------------------------------------
        # Play outcome
        # -----------------------------------------------------
        "yards_gained",
        "penalty",
        "penalty_team",

        # -----------------------------------------------------
        # Drive outcome
        # -----------------------------------------------------
        "drive_result",
        "drive_result_id",
        "drive_success",

        # -----------------------------------------------------
        # Absorbing-state variables
        # -----------------------------------------------------
        "td_team",
        "field_goal_result",
        "punt_attempt",
        "fumble_lost",
        "interception",
        "safety",
        "fourth_down_failed",
        "quarter_end",
        "absorbing_state",
        "absorbing_state_id",

        # -----------------------------------------------------
        # Game state / time
        # -----------------------------------------------------
        "qtr",
        "quarter_seconds_remaining",
        "half_seconds_remaining",
        "game_seconds_remaining",

        "posteam_score",
        "defteam_score",
        "score_differential",

        # -----------------------------------------------------
        # Covariates
        # -----------------------------------------------------
        "ep",
        "wp",
        "vegas_wp",

        "spread_line",
        "total_line",

        "surface",
        "temp",
        "wind",
        "weather"
    ]

    # Only retain columns that exist in the source dataframe
    cols_to_keep = [
        col for col in cols_to_keep
        if col in df.columns
    ]

    return df[cols_to_keep].copy()

def prepare_drive_level_data(plays: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate play-level NFL data into one row per offensive drive.

    Assumes `plays` has already been cleaned and validated and contains
    the following relevant columns:
        - game/drive identifiers
        - state information
        - absorbing_state
        - field position
        - scoring information
        - time/game context
        - external game context
    """

    # Ensure plays are in chronological order
    plays = plays.sort_values(
        [
            "game_date",
            "game_id",
            "drive_number",
            "play_number"
        ]
    ).copy()

    # ---------------------------------------------------------
    # First and last play of each drive
    # ---------------------------------------------------------

    grouped = plays.groupby(
        ["game_id", "drive_number"],
        sort=False
    )

    first = grouped.first().reset_index()
    last = grouped.last().reset_index()

    num_plays = (
        grouped.size()
        .rename("num_plays")
        .reset_index()
    )

    # ---------------------------------------------------------
    # Start-state information
    # ---------------------------------------------------------

    df = first[
        [
            "game_id",
            "drive_number",
            "season",
            "season_type",
            "week",
            "game_date",
            "posteam",
            "defteam",
            "posteam_type",

            "cur_state",
            "cur_state_id",

            "field_position",
            "binned_field_position",
            "binned_field_position_id",

            "game_seconds_remaining",
            "qtr",

            "posteam_score",
            "defteam_score",
            "score_differential",

            "spread_line",
            "total_line",
            "surface",
            "temp",
            "wind",
            "weather",
        ]
    ].copy()

    df = df.rename(
        columns={
            "cur_state": "start_state",
            "cur_state_id": "start_state_id",
            "field_position": "start_field_position",
            "binned_field_position": "start_binned_field_position",
            "binned_field_position_id": "start_binned_field_position_id",
            "game_seconds_remaining": "start_time_rem",
            "qtr": "start_qtr",
            "posteam_score": "start_posteam_score",
            "defteam_score": "start_defteam_score",
            "score_differential": "start_score_differential",
        }
    )

    # ---------------------------------------------------------
    # End-state / outcome information
    # ---------------------------------------------------------

    end = last[
        [
            "game_id",
            "drive_number",
            "field_position",
            "binned_field_position",
            "binned_field_position_id",
            "game_seconds_remaining",
            "qtr",
            "drive_result",
            "drive_result_id",
        ]
    ].copy()

    end = end.rename(
        columns={
            "field_position": "end_field_position",
            "binned_field_position": "end_binned_field_position",
            "binned_field_position_id": "end_binned_field_position_id",
            "game_seconds_remaining": "end_time_rem",
            "qtr": "end_qtr",
        }
    )

    # ---------------------------------------------------------
    # Combine start and end information
    # ---------------------------------------------------------

    df = df.merge(
        end,
        on=["game_id", "drive_number"],
        how="inner"
    )

    df = df.merge(
        num_plays,
        on=["game_id", "drive_number"],
        how="left"
    )

    # ---------------------------------------------------------
    # Drive outcome
    # ---------------------------------------------------------

    df["points"] = (
        df["drive_result"]
        .map({
            "TD": 7,
            "FG": 3,
            "PUNT": 0,
            "HALF_END": 0,
            "TURNOVER": 0,
            "DOWNS": 0,
        })
        .fillna(0)
        .astype(int)
    )

    df["scored"] = df["points"].gt(0)
    df["td_in_drive"] = df["drive_result"].eq("TD")
    df["fg_in_drive"] = df["drive_result"].eq("FG")

    # ---------------------------------------------------------
    # Field-position change
    # ---------------------------------------------------------

    df["net_field_position_change"] = df["start_field_position"] - df["end_field_position"]

    # ---------------------------------------------------------
    # Time
    # ---------------------------------------------------------

    df["time_elapsed"] = df["start_time_rem"] - df["end_time_rem"]

    # ---------------------------------------------------------
    # Final column order
    # ---------------------------------------------------------

    cols = [
        "season",
        "season_type",
        "week",
        "game_id",
        "game_date",
        "drive_number",
        "posteam",
        "defteam",
        "posteam_type",

        "start_state",
        "start_state_id",

        "start_field_position",
        "start_binned_field_position",
        "start_binned_field_position_id",

        "drive_result",
        "drive_result_id",
        "points",
        "scored",
        "td_in_drive",
        "fg_in_drive",

        "end_field_position",
        "end_binned_field_position",
        "end_binned_field_position_id",

        "net_field_position_change",

        "num_plays",

        "start_time_rem",
        "end_time_rem",
        "time_elapsed",

        "start_qtr",
        "end_qtr",

        "start_posteam_score",
        "start_defteam_score",
        "start_score_differential",

        "spread_line",
        "total_line",
        "surface",
        "temp",
        "wind",
        "weather",
    ]

    return df[cols].reset_index(drop=True)

def create_clean_pbp_files(
    years=list(range(2006,2026)),
    save_intermediates=True,
    save_dir=PBP_DIR,
    overwrite=False
):
    print('Creating clean play-by-play CSV files')

    full_csv_path = os.path.join(save_dir, f'pbp_{min(years)}_{max(years)}_full.csv')
    if not os.path.exists(full_csv_path) or (save_intermediates and overwrite):
        dfs = [
            pd.read_parquet(
                os.path.join(PBP_DIR, f'play_by_play_{year}.parquet'),
                engine='pyarrow'
            )
            for year in years
        ]
        df_full = pd.concat(dfs)

        if save_intermediates:
            print(f'Saving full CSV for years: {years}')
            df_full.to_csv(full_csv_path, index=False)
    else:
        print(f'Loading full CSV for years: {years}')
        df_full = pd.read_csv(full_csv_path)

    cols_to_keep = [
        'play_id', 'game_id', 'season_type', 'week', 'posteam', 'posteam_type', 'defteam',
        'yardline_100', 'game_date', 'drive', 'qtr', 'down', 'ydstogo', 
        'yards_gained', 'field_goal_result', 'td_team', 'punt_attempt', 'quarter_end',
        'fumble_lost', 'interception', 'safety', 'fourth_down_failed',
        'posteam_score', 'defteam_score', 'score_differential', 'season', 'penalty', 'penalty_team',
        'ep', 'wp', 'vegas_wp', 'spread_line', 'total_line', 'surface', 'temp', 'wind', "weather",
        'quarter_seconds_remaining', 'half_seconds_remaining', 'game_seconds_remaining', "desc"
    ]

    sub_csv_path = os.path.join(save_dir, f'pbp_{min(years)}_{max(years)}_sub.csv')
    if not os.path.exists(sub_csv_path) or (save_intermediates and overwrite):
        df_sub = df_full[cols_to_keep]

        if save_intermediates:
            print(f'Saving cropped CSV for years: {years}')
            df_sub.to_csv(sub_csv_path, index=False)

    else:
        print(f'Loading cropped CSV for years: {years}')
        df_sub = pd.read_csv(sub_csv_path)

    play_csv_path = os.path.join(save_dir, f'pbp_{min(years)}_{max(years)}_plays.csv')
    if overwrite or not os.path.exists(play_csv_path):
        df_play_clean = prepare_play_level_data(df_sub)
        print(f'Saving play-level CSV for years: {years}')
        df_play_clean.to_csv(play_csv_path, index=False)
    else:
        print(f'Loading play-level CSV for years: {years}')
        df_play_clean = pd.read_csv(play_csv_path)

    print(df_play_clean.shape)

    drive_csv_path = os.path.join(save_dir, f'pbp_{min(years)}_{max(years)}_drives.csv')
    if overwrite or not os.path.exists(drive_csv_path):
        df_drive_clean = prepare_drive_level_data(df_play_clean)
        print(f'Saving drive-level CSV for years: {years}')
        df_drive_clean.to_csv(drive_csv_path, index=False)
    else:
        print(f'Loading drive-level CSV for years: {years}')
        df_drive_clean = pd.read_csv(drive_csv_path)

    print(df_drive_clean.shape)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Create cleaned NFL play-by-play CSVs from yearly parquet files.'
    )

    parser.add_argument(
        '--years', "-yrs", '-y',
        nargs='+',
        type=int,
        default=list(range(2006, 2026)),
        help='Years to process. Defaults to 2006 through 2025.'
    )
    parser.add_argument(
        '--no-intermediates', '--no-int',
        action='store_true',
        default=False,
        help='Do not save intermediate CSV files.'
    )
    parser.add_argument(
        '--save-dir', '-sdir',
        type=str,
        default=PBP_DIR,
        help=f'Directory where CSVs will be saved. Defaults to {PBP_DIR}.'
    )
    parser.add_argument(
        '--overwrite', '-ow',
        action='store_true',
        default=False,
        help='Overwrite existing CSV files.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    create_clean_pbp_files(
        years=args.years,
        save_intermediates=not args.no_intermediates,
        save_dir=args.save_dir,
        overwrite=args.overwrite,
    )


