import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# creating a function to calculate the win/loss ratio for each team given the seasonal data
def win_loss (schedule_df): 
    """
    This function calculates the win/loss ratio for each team in the schedule dataframe
    """
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


# getting fantasy defense ranking through week by week data
def fantasy_defense_rankings(week_by_week_df): 
    """
    This function calculates the fantasy defense rankings for each team based on the fantasy points scored against them
    """
    smallFantasy_df = week_by_week_df[['fantasy_points_ppr', 'position', 'opponent_team', 'player_name', 'week']]
    releventPositions_df = smallFantasy_df[smallFantasy_df['position'].isin(['QB', 'RB', 'WR', 'TE'])]
    
    # using top performers b/c thats usually more indicative of sos in fantasy football 
    topPerformers_df = releventPositions_df.sort_values(by='fantasy_points_ppr', ascending=False).groupby(['opponent_team', 'position']).head(10) # can use .head() if i want to see top performer averages
    
    
    # groupy by opponent team and position in order to find out how many total fantasy points each position has scored against them 
    defensePointsAgainst = topPerformers_df.groupby(['opponent_team', 'position'])['fantasy_points_ppr'].mean().reset_index()
    
    # assign a rank to each team based on the total fantasy points scored against them
    defensePointsAgainst['rank'] = defensePointsAgainst.groupby('position')['fantasy_points_ppr'].rank(ascending=True)
    
    defensePointsAgainst = defensePointsAgainst.sort_values(by=['position', 'rank'])
    
    defenseRankings = defensePointsAgainst.pivot(index = "opponent_team", columns = "position", values = "rank")
    
    defensePPRPoints = defensePointsAgainst.pivot(index = "opponent_team", columns = "position", values = "fantasy_points_ppr")
    
    defenseRankings = defenseRankings.reset_index()
    defensePPRPoints = defensePPRPoints.reset_index()
    
    defenseRankings = defenseRankings.merge(defensePPRPoints, on = "opponent_team", suffixes = ("_rank", "_ppr_avg"))

    
    return defenseRankings


def merge_defense_schedule (schedule_df, defense_df): 
    """
    Merges the defense rankings with the schedule dataframe
    """
    # merging the defense rankings with the schedule dataframe
    home_df = defense_df.add_suffix("_home")
    scheduleWithDefense = schedule_df.merge(home_df, left_on="home_team", right_on="opponent_team_home", how="left", suffixes=("", "_home"))
    scheduleWithDefense = scheduleWithDefense.merge(defense_df, left_on="away_team", right_on="opponent_team", how="left", suffixes=("", "_away"))
    scheduleWithDefense = scheduleWithDefense.drop(columns=["opponent_team_home", "opponent_team_away"], errors="ignore")
    return scheduleWithDefense