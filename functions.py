import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# creating a function to calculate the win/loss ratio for each team given the seasonal data
def win_loss (schedule_df): 
    # making the dataframe smaller so its easier to handle the data
    smallSchedule2024 = schedule_df[['result', 'away_team', 'away_score', 'home_team', 'home_score', 'week']]   
    previousGames = smallSchedule2024[schedule_df['result'].notnull()]
    
    
    previousGames['home_win'] = np.where(previousGames['result'] > 0, 1, 0)
    previousGames['away_win'] = np.where(previousGames['result'] < 0, 1, 0)
    
    ## not sure if all this is necessary. I think the best way to do it would be to just count the 1s and 0s to figure out the record
    # obtaining the week list
    weekList = previousGames['week'].unique()
    
    #going week by week to calculate the records
    for week in weekList: 
        # getting the data for each week at a time (to tally up the win/loss at the specific time of playing)
        previousWeekData = previousGames[previousGames['week'] <= week]
        
        # calculating the wins and losses for each team
        homeWins = previousWeekData.groupby('home_team')['home_win'].sum()
        awayWins = previousWeekData.groupby('away_team')['away_win'].sum()
        homeLosses = previousWeekData.groupby('home_team')['home_win'].count() - previousWeekData.groupby('home_team')['home_win'].sum()
        awayLosses = previousWeekData.groupby('away_team')['away_win'].count() - previousWeekData.groupby('away_team')['away_win'].sum()
        
        teamWins = homeWins.add(awayWins, fill_value=0)
        teamLosses = homeLosses.add(awayLosses, fill_value=0)
        
        # creates a mask so we get the right week in the origianl df
        mask = schedule_df['week'] == week

        # adding the win/loss record to the schedule_df
        schedule_df.loc[mask, 'home_team_win_record'] = schedule_df['home_team'].map(teamWins)
        schedule_df.loc[mask, 'home_team_loss_record'] = schedule_df['home_team'].map(teamLosses)
        schedule_df.loc[mask, 'away_team_win_record'] = schedule_df['away_team'].map(teamWins)
        schedule_df.loc[mask, 'away_team_loss_record'] = schedule_df['away_team'].map(teamLosses)

    return schedule_df
