import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

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

def get_team_schedule (schedule_df,team_name, start_week = 1 , end_week = 18): 
    # filter rows where the team is either the home or away team
    team_home_games = schedule_df[schedule_df['home_team'] == team_name]
    team_away_games = schedule_df[schedule_df['away_team'] == team_name]

    # create lists of matchups
    home_games = [row['away_team'] for _, row in team_home_games.iterrows()]
    away_games = [row['home_team'] for _, row in team_away_games.iterrows()]

   # create a full schedule for the team
    home_schedule = team_home_games[['week']].assign(matchup=home_games)
    away_schedule = team_away_games[['week']].assign(matchup=away_games)
    full_schedule = pd.concat([home_schedule, away_schedule], ignore_index=True).sort_values(by='week').reset_index(drop=True)
   
    # identify bye weeks (weeks where the team is not playing)
    all_weeks = set(range(1, 18))
    played_weeks = set(full_schedule['week'])
    bye_weeks = list(all_weeks - played_weeks)

    # add bye weeks to the schedule
    bye_schedule = pd.DataFrame({"week": bye_weeks, "matchup": ["BYE"] * len(bye_weeks)})
    full_schedule = pd.concat([full_schedule, bye_schedule], ignore_index=True).sort_values(by='week').reset_index(drop=True)

    schedule_list = full_schedule['matchup'].tolist()

    # account for a specific range of weeks
    sorted_schedule_list = schedule_list[(start_week - 1):(end_week)]

    return sorted_schedule_list

# creating a function to calculate the average fantasy points scored against each team for the rest of the season
def get_defense_matchups (schedule_df, week_by_week_df, start_week = 1, end_week = 18): 
    """
    This function calculates the average fantasy points scored against each team for the rest of the season
    """
    defenseRankings_df = fantasy_defense_rankings(week_by_week_df).set_index("opponent_team")
    
    # creating a dataframe with the team and their matchups for the rest of the season
    matchups_data = []
    listOfTeams = week_by_week_df["recent_team"].unique()
    
    for team in listOfTeams:
        team_matchups = get_team_schedule(schedule_df, team, start_week=start_week, end_week=end_week)

        # precompute opponent ranks for each position using dictionary lookups
        qb_opponent_ranks = [defenseRankings_df.at[opponent, "QB_rank"] if opponent in defenseRankings_df.index else np.nan for opponent in team_matchups]
        rb_opponent_ranks = [defenseRankings_df.at[opponent, "RB_rank"] if opponent in defenseRankings_df.index else np.nan for opponent in team_matchups]
        wr_opponent_ranks = [defenseRankings_df.at[opponent, "WR_rank"] if opponent in defenseRankings_df.index else np.nan for opponent in team_matchups]
        te_opponent_ranks = [defenseRankings_df.at[opponent, "TE_rank"] if opponent in defenseRankings_df.index else np.nan for opponent in team_matchups]

        # calculate the average ranks
        qb_opponent_average = np.nanmean(qb_opponent_ranks)
        rb_opponent_average = np.nanmean(rb_opponent_ranks)
        wr_opponent_average = np.nanmean(wr_opponent_ranks)
        te_opponent_average = np.nanmean(te_opponent_ranks)
        
        # append to the matchups list
        matchups_data.append({
            "team": team,
            "matchups": team_matchups,
            "qb_opponent_average": qb_opponent_average,
            "rb_opponent_average": rb_opponent_average,
            "wr_opponent_average": wr_opponent_average,
            "te_opponent_average": te_opponent_average
        })

    matchups_df = pd.DataFrame(matchups_data)
    
    return matchups_df


def calculate_top_player_matchup(schedule_df, week_by_week_df, start_week=1, end_week=18): 
    
    defenseRankings_df = fantasy_defense_rankings(week_by_week_df).set_index("opponent_team")
    
    # creating a dataframe with the team and their matchups for the rest of the season
    matchups_data = []
    listOfTeams = week_by_week_df["recent_team"].unique()
    
    for team in listOfTeams:
        team_matchups = get_team_schedule(schedule_df, team, start_week, end_week)
        
        # precompute opponent ranks for each position using dictionary lookups
        qb_opponent_ranks = [defenseRankings_df.at[opponent, "QB_rank"] if opponent in defenseRankings_df.index else np.nan for opponent in team_matchups]
        rb_opponent_ranks = [defenseRankings_df.at[opponent, "RB_rank"] if opponent in defenseRankings_df.index else np.nan for opponent in team_matchups]
        wr_opponent_ranks = [defenseRankings_df.at[opponent, "WR_rank"] if opponent in defenseRankings_df.index else np.nan for opponent in team_matchups]
        te_opponent_ranks = [defenseRankings_df.at[opponent, "TE_rank"] if opponent in defenseRankings_df.index else np.nan for opponent in team_matchups]

        # calculate the average ranks
        qb_opponent_average = np.nanmean(qb_opponent_ranks)
        rb_opponent_average = np.nanmean(rb_opponent_ranks)
        wr_opponent_average = np.nanmean(wr_opponent_ranks)
        te_opponent_average = np.nanmean(te_opponent_ranks)
        
        # append to the matchups list
        matchups_data.append({
            "team": team,
            "matchups": team_matchups,
            "qb_opponent_average": qb_opponent_average,
            "rb_opponent_average": rb_opponent_average,
            "wr_opponent_average": wr_opponent_average,
            "te_opponent_average": te_opponent_average
        })

    matchups_df = pd.DataFrame(matchups_data)
         
    #trying to filter out the injured players
    latest_week = week_by_week_df["week"].max()
    recent_games_df = week_by_week_df[
    week_by_week_df["week"].isin([latest_week - 2, latest_week - 1, latest_week])
]
    active_players = recent_games_df["player_name"].unique().tolist()
    active_players.append("N.Collins")

    fantasyAverage_df = (
    week_by_week_df.groupby(["player_name", "position"])
    .agg({"fantasy_points_ppr": "mean", "recent_team": "last"})
    .reset_index()
    .sort_values("fantasy_points_ppr", ascending=False)
)
    fantasyAverage_df = fantasyAverage_df[fantasyAverage_df["player_name"].isin(active_players)]
    # getting the top 30 fantasy performers of each of three positions 
    top10_qb = fantasyAverage_df[fantasyAverage_df["position"] == "QB"].head(15)
    top30_rb = fantasyAverage_df[fantasyAverage_df["position"] == "RB"].head(35)
    top30_wr = fantasyAverage_df[fantasyAverage_df["position"] == "WR"].head(35)
    top12_te = fantasyAverage_df[fantasyAverage_df["position"] == "TE"].head(15)
    
    # if necessary i can get full weekbyweek data (this would be better for calculating a composite score and factor in other factors)
    # top_rb_weekly = week_by_week_df[week_by_week_df['player_name'].isin(top_rb["player_name"])]
    # top_wr_weekly = week_by_week_df[week_by_week_df['player_name'].isin(top_wr["player_name"])]
    # top_te_weekly = week_by_week_df[week_by_week_df['player_name'].isin(top_te["player_name"])]
    
    # merge the top30 rb with the matchups_df
    top30_rb_matchups = top30_rb.merge(matchups_df, left_on="recent_team", right_on="team", how="left")
    top30_wr_matchups = top30_wr.merge(matchups_df, left_on="recent_team", right_on="team", how="left")
    top12_te_matchups = top12_te.merge(matchups_df, left_on="recent_team", right_on="team", how="left")
    top10_qb_matchups = top10_qb.merge(matchups_df, left_on="recent_team", right_on="team", how="left")
    
    top10_qb_matchups = top10_qb_matchups.rename(columns={"fantasy_points_ppr": "avg_fantasy_points_ppr"})
    top30_rb_matchups = top30_rb_matchups.rename(columns={"fantasy_points_ppr": "avg_fantasy_points_ppr"})
    top30_wr_matchups = top30_wr_matchups.rename(columns={"fantasy_points_ppr": "avg_fantasy_points_ppr"})
    top12_te_matchups = top12_te_matchups.rename(columns={"fantasy_points_ppr": "avg_fantasy_points_ppr"})
    return top10_qb_matchups, top30_rb_matchups, top30_wr_matchups, top12_te_matchups
    
def plot_fantasy_vs_matchup(data_df, position, playoffs= True, filepath = "fantasy_vs_matchup.png"):
    plt.figure(figsize=(12, 8))

    # create the scatter plot
    scatter = plt.scatter(
        data_df["avg_fantasy_points_ppr"],
        data_df[f"{position.lower()}_opponent_average"],
        alpha=0.7
    )

    # add text annotations for each player
    texts = [
        plt.text(
            row["avg_fantasy_points_ppr"],
            row[f"{position.lower()}_opponent_average"],
            row["player_name"],
            fontsize=10
        )
        for index, row in data_df.iterrows()
    ]

    # adjust the text labels to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='black', lw=0.5))

     # calculate the plot limits
    min_fantasy_points = data_df["avg_fantasy_points_ppr"].min()
    max_fantasy_points = data_df["avg_fantasy_points_ppr"].max()
    min_opponent_rank = data_df[f"{position.lower()}_opponent_average"].min()
    max_opponent_rank = data_df[f"{position.lower()}_opponent_average"].max()

    # draw a diagonal line from top left (easiest matchups, low PPG) to bottom right (hardest matchups, high PPG)
    plt.plot(
        [min_fantasy_points, max_fantasy_points],
        [max_opponent_rank, min_opponent_rank],   
        color='purple',
        linestyle='--',
        label='Diagonal Split Line'
    )

    plt.legend()
    
    # labels and title
    plt.xlabel("Average PPR Fantasy Points")
    plt.ylabel("Playoff Average Opponent Matchup Rank (1 = Toughest Defense, 32 = Easiest Defense)")
    if playoffs:
        plt.title(f"Fantasy Points vs. Average Opponent Matchup Rank (Playoffs) for Top {position}s")
    else:
        plt.title(f"Fantasy Points vs. Average Opponent Matchup Rank (Rest of Season) for Top {position}s")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"visuals/{filepath}", dpi = 300)
    plt.show()
    
def zscore_and_weight(df, metric, weight):
    z_score = (df[metric] - df[metric].mean()) / df[metric].std()
    return z_score * weight

def getOffensiveQuality(week_by_week_df):
    # calculate each metric
    week_by_week_df['total_yards'] = week_by_week_df['passing_yards'] + week_by_week_df['rushing_yards']
    week_by_week_df['total_first_downs'] = week_by_week_df['passing_first_downs'] + week_by_week_df['rushing_first_downs']
    week_by_week_df['total_touchdowns'] = week_by_week_df['rushing_tds'] + week_by_week_df['passing_tds']
    
    # aggregate season totals per team
    offensive_quality = week_by_week_df.groupby(['recent_team']).agg(
        total_yards=('total_yards', 'sum'),
        total_first_downs=('total_first_downs', 'sum'),
        total_touchdowns=('total_touchdowns', 'sum')
    )
    
    # calculate z scores for each metric
    offensive_quality['z_yards'] = (offensive_quality['total_yards'] - offensive_quality['total_yards'].mean()) / offensive_quality['total_yards'].std()
    offensive_quality['z_first_downs'] = (offensive_quality['total_first_downs'] - offensive_quality['total_first_downs'].mean()) / offensive_quality['total_first_downs'].std()
    offensive_quality['z_touchdowns'] = (offensive_quality['total_touchdowns'] - offensive_quality['total_touchdowns'].mean()) / offensive_quality['total_touchdowns'].std()
    
    # average z scores to get a composite offensive quality score
    offensive_quality['offensive_quality_score'] = .275*offensive_quality['z_yards'] + .275*offensive_quality['z_first_downs'] +  .45* offensive_quality['z_touchdowns']
    
    return offensive_quality[['offensive_quality_score']]



# creating a composite score, going to factor in : 
# TE : target_share, air_yard_target_share, receiving_epa, fantasy_points_ppr, opponent_rank 
# wr : wopr, receiving_epa, fantasy_points_ppr, opponent_rank
# rb: carries,  targets, rushing_tds + receiving_tds, rushing_epa, fantasy_points_ppr, opponent_rank
#also factor in PPG_DROP 
#
def compositeBuyLowRankings (weekByWeek_df, schedule_df): 
    offensiveQuality_df = getOffensiveQuality(weekByWeek_df)
    top10_qb, top35_rb, top35_wr, top12_te = calculate_top_player_matchup(schedule_df, weekByWeek_df, 15, 17) # obtaining the best matchups for playoff weeks
    
    top_rb_weekByWeek = weekByWeek_df[weekByWeek_df['player_name'].isin(top35_rb["player_name"])]
    top_wr_weekByWeek = weekByWeek_df[weekByWeek_df['player_name'].isin(top35_wr["player_name"])]
    top_te_weekByWeek = weekByWeek_df[weekByWeek_df['player_name'].isin(top12_te["player_name"])]
    
    top_rb_weekByWeek["total_touchdowns"] = top_rb_weekByWeek["rushing_tds"] + top_rb_weekByWeek["receiving_tds"]
    
    # obtaining all the metrics I want to use
    topRBStats = top_rb_weekByWeek.groupby(['player_name']).agg(
        carries=('carries', 'mean'),
        targets=('targets', 'mean'),
        total_tds=('total_touchdowns', 'mean'),
        avg_rushing_epa=('rushing_epa', 'mean'), 
        recent_team = ('recent_team', 'last'),
    ).reset_index()
    topWRStats = top_wr_weekByWeek.groupby(['player_name']).agg(
        wopr=('wopr', 'mean'),
        avg_receiving_epa=('receiving_epa', 'mean'),
        recent_team = ('recent_team', 'last'),
    ).reset_index()
    topTEStats = top_te_weekByWeek.groupby(['player_name']).agg(
        target_share=('target_share', 'mean'),
        air_yard_target_share=('air_yards_share', 'mean'),
        avg_receiving_epa=('receiving_epa', 'mean'),
        recent_team = ('recent_team', 'last'),
    ).reset_index()
    
    #merging all the datafarmes
    topRBStats = topRBStats.merge(offensiveQuality_df, left_on='recent_team', right_on='recent_team', how='left')
    topRBStatsWithMatchups = topRBStats.merge(top35_rb, left_on='player_name', right_on='player_name', how='left')
    topWRStats = topWRStats.merge(offensiveQuality_df, left_on='recent_team', right_on='recent_team', how='left')
    topWRStatsWithMatchups = topWRStats.merge(top35_wr, left_on='player_name', right_on='player_name', how='left')
    topTEStats = topTEStats.merge(offensiveQuality_df, left_on='recent_team', right_on='recent_team', how='left')
    topTEStatsWithMatchups = topTEStats.merge(top12_te, left_on='player_name', right_on='player_name', how='left')
    
    # calculate PPG Drop 
    def calculate_ppg_drop(df, weekByWeek_df):
        last_2_games = weekByWeek_df.groupby('player_name').tail(2)
        last_2_games_avg = last_2_games.groupby('player_name')['fantasy_points_ppr'].mean()

        season_avg = df.set_index('player_name')['avg_fantasy_points_ppr']

        ppg_drop = season_avg.subtract(last_2_games_avg, fill_value=0)
        df['ppg_drop'] = df['player_name'].map(ppg_drop).fillna(0)
        df['ppg_drop'] = df['ppg_drop'].apply(lambda x: round(x, 2))
        return df

    # add PPG Drop to each stats DataFrame
    topRBStatsWithMatchups = calculate_ppg_drop(topRBStatsWithMatchups, top_rb_weekByWeek)
    topWRStatsWithMatchups = calculate_ppg_drop(topWRStatsWithMatchups, top_wr_weekByWeek)
    topTEStatsWithMatchups = calculate_ppg_drop(topTEStatsWithMatchups, top_te_weekByWeek)
    
    
    #z standardizing everything!!
    rbMetrics = ['carries', 'targets', 'total_tds', 'avg_rushing_epa', "offensive_quality_score", "rb_opponent_average", "avg_fantasy_points_ppr"]
    for metric in rbMetrics: 
        topRBStatsWithMatchups[f'z_{metric}'] = (topRBStatsWithMatchups[metric] - topRBStatsWithMatchups[metric].mean()) / topRBStatsWithMatchups[metric].std()
    wrMetrics = ['wopr', 'avg_receiving_epa',"offensive_quality_score", "wr_opponent_average", 'avg_fantasy_points_ppr']
    for metric in wrMetrics:
        topWRStatsWithMatchups[f'z_{metric}'] = (topWRStatsWithMatchups[metric] - topWRStatsWithMatchups[metric].mean()) / topWRStatsWithMatchups[metric].std()
    TEMetrics= ['target_share', 'air_yard_target_share', 'avg_receiving_epa', "offensive_quality_score", "te_opponent_average", 'avg_fantasy_points_ppr']
    for metric in TEMetrics:
        topTEStatsWithMatchups[f'z_{metric}'] = (topTEStatsWithMatchups[metric] - topTEStatsWithMatchups[metric].mean()) / topTEStatsWithMatchups[metric].std()
    
    #creating objects with the weights for each metric
    rb_weights = {
        'z_carries': 0.1,
        'z_targets': 0.15,
        'z_total_tds': 0.2,
        'z_avg_rushing_epa': 0.05,
        'z_offensive_quality_score': 0.2,
        'z_avg_fantasy_points_ppr': 0.2,
        'z_rb_opponent_average': 0.1
    }
    wr_weights = {
        'z_wopr': 0.3,
        'z_avg_receiving_epa': 0.2,
        'z_offensive_quality_score': 0.2,
        'z_avg_fantasy_points_ppr': 0.2,
        'z_wr_opponent_average': 0.1
    }
    te_weights = {
        'z_target_share': 0.2,
        'z_air_yard_target_share': 0.2,
        'z_avg_receiving_epa': 0.2,
        'z_offensive_quality_score': 0.1,
        'z_avg_fantasy_points_ppr': 0.2, 
        'z_te_opponent_average': 0.1
    }
    
    def calculate_composite_score(df, weights):
        composite_score = 0
        for metric, weight in weights.items():
            composite_score += df[metric] * weight
            
            composite_score = round(composite_score, 2)
        return composite_score
    
    topRBStatsWithMatchups['composite_score'] = calculate_composite_score(topRBStatsWithMatchups, rb_weights)
    topWRStatsWithMatchups['composite_score'] = calculate_composite_score(topWRStatsWithMatchups, wr_weights)
    topTEStatsWithMatchups['composite_score'] = calculate_composite_score(topTEStatsWithMatchups, te_weights)
    
    topRBStatsWithMatchups['rank'] = topRBStatsWithMatchups['composite_score'].rank(ascending=False)
    topWRStatsWithMatchups['rank'] = topWRStatsWithMatchups['composite_score'].rank(ascending=False)
    topTEStatsWithMatchups['rank'] = topTEStatsWithMatchups['composite_score'].rank(ascending=False)
    
    
    return topRBStatsWithMatchups[['player_name', 'composite_score', 'rank', 'ppg_drop']].sort_values(by='ppg_drop', ascending =False), topWRStatsWithMatchups[['player_name', 'composite_score', 'rank', 'ppg_drop']].sort_values(by='ppg_drop', ascending =False), topTEStatsWithMatchups[['player_name', 'composite_score', 'rank', 'ppg_drop']].sort_values(by='ppg_drop', ascending =False)

def create_table_screenshot(df, file_path="table_screenshot.png", max_rows=15):
    """
    Creates a screenshot of a DataFrame as a table and saves it as an image.
    """
    # limit the number of rows for better readability
    if len(df) > max_rows:
        df = df.head(max_rows)

    fig, ax = plt.subplots(figsize=(12, len(df) * 0.5))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    for i, key in enumerate(df.columns):
        table[0, i].set_facecolor("#40466e")
        table[0, i].set_text_props(color="w", weight="bold")
    for j in range(1, len(df) + 1):
        if j % 2 == 0:
            for i in range(len(df.columns)):
                table[j, i].set_facecolor("#f0f0f0")

    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.tight_layout()
    plt.close()