# nfl-playoffs-predictor

## Final Stats Data

The `final_stats/` directory contains four CSV files:

- `weekly_stats_4.csv`
- `weekly_stats_6.csv`
- `weekly_stats_10.csv`
- `playoffs_2021_to_2024.csv`

### Weekly Stats Files

Each `weekly_stats_*.csv` file provides weekly offensive and defensive statistics for NFL teams from the last four regular seasons (2021–2024). The number in the filename indicates the cumulative number of games included:

- **4**: Stats up to each team's first 4 games of the season.
- **6**: Stats up to each team's first 6 games of the season.
- **10**: Stats up to each team's first 10 games of the season.

#### What is included?

Each row in these files represents a team's performance in a given week, including metrics such as:

- Turnover margin
- Offensive and defensive efficiency ratios
- Touchdown efficiency
- Points scored and allowed
- Total, passing, and rushing yards
- First downs, turnovers, sacks, and more

These stats are useful for analyzing team performance trends early in the season and for building predictive models for playoff outcomes.

**Note:** The stats show team's weekly stats up to the first n weeks (e.g., week 10 in `weekly_stats_10.csv` means stats from the first 10 games regardless of the bye week)

### Playoff Results File

`playoffs_2021_to_2024.csv` contains playoff results for the 2021–2024 NFL seasons.
