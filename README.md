# nfl-playoffs-predictor

## Final Stats Data

The `final_stats/` directory contains three CSV files:

- `weekly_stats_4.csv`
- `weekly_stats_6.csv`
- `weekly_stats_10.csv`

Each file provides weekly offensive and defensive statistics for NFL teams from the last four regular seasons (2020-2024). The difference between the files is the number in their name:

- **4**: Stats accumulated up to each team's first 4 games of the season.
- **6**: Stats accumulated up to each team's first 6 games of the season.
- **10**: Stats accumulated up to each team's first 10 games of the season.

### What is included?

Each row in these files represents a team's performance in a given week, including metrics such as:

- Turnover margin
- Offensive and defensive efficiency ratios
- Touchdown efficiency
- Points scored and allowed
- Total, passing, and rushing yards
- First downs, turnovers, sacks, and more

These stats are useful for analyzing team performance trends early in the season and for building predictive models for playoff outcomes.

**Note:** The stats are cumulative up to the specified week (e.g., week 4 in `weekly_stats_4.csv` means stats from weeks 1–4 combined).
