from src.State import State
import pandas as pd

def sim_drive(
        start_state,
        model
    ) -> list:
    curr_state = start_state
    drive = []
    drive.append(curr_state)
    # yds_gained_list = []
    while curr_state.down != 4:
        curr_play_dict = {"yardline_100": [curr_state.yardline_100], "ydstogo": [curr_state.ydstogo], "down": [curr_state.down]}
        for key, val in start_state.covars.items():
            curr_play_dict[key] = [val]
        curr_play_for_model = pd.DataFrame(curr_play_dict)
        yds_gained = model.predict(curr_play_for_model)[0]
        # yds_gained_list.append(yds_gained)
        new_down = 0
        new_yds_to_go = 0
        if yds_gained >= curr_state.ydstogo and yds_gained < curr_state.yardline_100:
            new_yds_to_go = 10
            new_down = 1
        elif yds_gained < curr_state.yardline_100:
            new_yds_to_go = curr_state.ydstogo - yds_gained
            new_down = curr_state.down + 1
        else:
            break
        new_state = State(new_down, new_yds_to_go, curr_state.yardline_100 - yds_gained, curr_state.covars)
        curr_state = new_state
        drive.append(curr_state)
    return drive
    # return {'drive' : drive, 'yds_gained' : yds_gained_list}