import pandas as pd

# Load data
df = pd.read_csv('data/processed/odds_processed.csv')

print("VERIFICATION: Always Bet Home Strategy")
print("=" * 70)

# Strategy: Bet $20 on every home team (2% of $1000)
stake_per_bet = 20
total_matches = len(df)

# Calculate profit/loss for each match
profits = []

for idx, row in df.iterrows():
    if row['FTR'] == 'H':  # Home win
        profit = stake_per_bet * (row['B365H'] - 1)
    else:  # Home loss
        profit = -stake_per_bet
    
    profits.append(profit)

df['bet_profit'] = profits

# Summary statistics
total_profit = df['bet_profit'].sum()
total_staked = total_matches * stake_per_bet
roi = (total_profit / total_staked) * 100

wins = (df['FTR'] == 'H').sum()
win_rate = wins / total_matches

print(f"\nTotal matches: {total_matches}")
print(f"Stake per bet: ${stake_per_bet}")
print(f"Total staked: ${total_staked:,.2f}")
print(f"\nHome wins: {wins} ({win_rate:.2%})")
print(f"Home losses: {total_matches - wins} ({(1-win_rate):.2%})")
print(f"\nTotal profit: ${total_profit:,.2f}")
print(f"ROI: {roi:.2f}%")
print(f"Final bankroll: ${1000 + total_profit:,.2f}")

# Show first 10 bets
print("\n" + "=" * 70)
print("FIRST 10 BETS DETAIL:")
print("=" * 70)
sample = df[['HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'bet_profit']].head(10)
print(sample.to_string())

# Check average odds for wins vs all
avg_odds_all = df['B365H'].mean()
avg_odds_wins = df[df['FTR'] == 'H']['B365H'].mean()

print("\n" + "=" * 70)
print(f"Average home odds (all matches): {avg_odds_all:.3f}")
print(f"Average home odds (when home wins): {avg_odds_wins:.3f}")
print("\nExpected value per bet:")
print(f"  Win: {win_rate:.3f} × ${stake_per_bet} × {avg_odds_wins:.3f} = ${win_rate * stake_per_bet * avg_odds_wins:.2f}")
print(f"  Loss: {1-win_rate:.3f} × $-{stake_per_bet} = $-{(1-win_rate) * stake_per_bet:.2f}")
print(f"  Net EV: ${win_rate * stake_per_bet * avg_odds_wins - (1-win_rate) * stake_per_bet:.2f}")