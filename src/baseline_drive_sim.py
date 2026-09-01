from src.State import State
import pandas as pd

def sim_drive(
        start_state,
        action_model,
        yards_gained_model
    ) -> list:
    """
    Funciton to simulate a drive
    Parameters:
        start_state: State class, gives the starting state (down, distance, yardline, etc.) of the drive
        action_model: sklearn.pipeline.Pipeline class, gives the action ('GO', 'TURNOVER', 'FG', 'PUNT') given down, distance, yardline
        yards_gained_model: sklearn.pipeline.Pipeline class, gives the yards gained given down, distance, yardline, and other covariates
    Returns:
        List of states representing the drive
    """
    curr_state = start_state
    drive = []
    drive.append(curr_state)

    def create_curr_play_for_model(curr_state):
        """
        Helper function to turn the state into a pandas dataframe for model prediction
        Parameters:
            curr_state: State class, the state to be transformed
        Returns:
            pd.DataFrame of the state 
        """
        curr_play_dict = {"yardline_100": [curr_state.yardline_100], "ydstogo": [curr_state.ydstogo], "down": [curr_state.down]}
        for key, val in curr_state.covars.items():
            curr_play_dict[key] = [val]
        curr_play_for_model = pd.DataFrame(curr_play_dict)
        return curr_play_for_model
    
    curr_play_for_model = create_curr_play_for_model(curr_state)
    #Predict the action
    action = action_model.predict(curr_play_for_model[['yardline_100', 'ydstogo', 'down']])[0]
    while action == 'GO':
        #Predict the yards gained
        yds_gained = yards_gained_model.predict(curr_play_for_model)[0]
        new_down = 0
        new_yds_to_go = 0
        #If it's 4th down and you don't get enough for a first, end drive
        if curr_state.down == 4 and yds_gained < curr_state.ydstogo:
            break
        #Otherwise, if you don't get enough for a first, update yards to go and increment down
        elif yds_gained < curr_state.ydstogo:
            new_yds_to_go = curr_state.ydstogo - yds_gained
            new_down = curr_state.down + 1
        #If you get enough for a first but not a touchdown, reset yards to go and down
        elif yds_gained >= curr_state.ydstogo and yds_gained < curr_state.yardline_100:
            new_yds_to_go = 10
            new_down = 1
        else:
            break
        #Create and append the new state
        new_state = State(new_down, new_yds_to_go, curr_state.yardline_100 - yds_gained, curr_state.covars)
        curr_state = new_state
        drive.append(curr_state)
        curr_play_for_model = create_curr_play_for_model(curr_state)
        #Predict the next action
        action = action_model.predict(curr_play_for_model[['yardline_100', 'ydstogo', 'down']])[0]
    return drive