import pandas as pd
import numpy as np

# Load processed data
df = pd.read_csv('data/processed/odds_processed.csv')

print("=" * 70)
print("DIAGNOSTIC CHECK")
print("=" * 70)

# 1. Check data shape and completeness
print("\n1. DATA OVERVIEW")
print(f"Total rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"\nKey columns present:")
print(f"  - FTR (results): {df['FTR'].notna().sum()}")
print(f"  - B365H (home odds): {df['B365H'].notna().sum()}")
print(f"  - B365D (draw odds): {df['B365D'].notna().sum()}")
print(f"  - B365A (away odds): {df['B365A'].notna().sum()}")

# 2. Check odds values
print("\n2. ODDS STATISTICS")
print(f"Average home odds: {df['B365H'].mean():.2f}")
print(f"Average draw odds: {df['B365D'].mean():.2f}")
print(f"Average away odds: {df['B365A'].mean():.2f}")
print(f"\nOdds ranges:")
print(f"  Home: {df['B365H'].min():.2f} - {df['B365H'].max():.2f}")
print(f"  Draw: {df['B365D'].min():.2f} - {df['B365D'].max():.2f}")
print(f"  Away: {df['B365A'].min():.2f} - {df['B365A'].max():.2f}")

# 3. Check outcome distribution
print("\n3. OUTCOME DISTRIBUTION")
print(df['FTR'].value_counts())
print(f"\nProportions:")
print(df['FTR'].value_counts(normalize=True))

# 4. Check probabilities
print("\n4. IMPLIED PROBABILITIES")
print(f"Average home prob: {df['B365_true_prob_H'].mean():.3f}")
print(f"Average draw prob: {df['B365_true_prob_D'].mean():.3f}")
print(f"Average away prob: {df['B365_true_prob_A'].mean():.3f}")
print(f"Sum (should be ~1.0): {(df['B365_true_prob_H'] + df['B365_true_prob_D'] + df['B365_true_prob_A']).mean():.3f}")

# 5. Sample some bets to verify logic
print("\n5. SAMPLE BET VERIFICATION")
print("\nFirst 5 matches with odds and outcomes:")
sample = df[['HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'B365D', 'B365A', 
             'B365_true_prob_H', 'B365_true_prob_D', 'B365_true_prob_A']].head(10)
print(sample.to_string())

# 6. Test a simple strategy manually
print("\n6. MANUAL STRATEGY TEST (Always bet home)")
total_bets = len(df)
home_wins = (df['FTR'] == 'H').sum()
avg_home_odds = df['B365H'].mean()

# Simulate: bet $1 on every home team
stake = 1
total_staked = total_bets * stake
total_returned = home_wins * stake * avg_home_odds
profit = total_returned - total_staked
roi = (profit / total_staked) * 100

print(f"Total bets: {total_bets}")
print(f"Home wins: {home_wins} ({home_wins/total_bets:.1%})")
print(f"Total staked: ${total_staked:.2f}")
print(f"Total returned: ${total_returned:.2f}")
print(f"Profit: ${profit:.2f}")
print(f"ROI: {roi:.2f}%")

# 7. Check for data issues
print("\n7. DATA QUALITY CHECKS")
print(f"Negative odds? {(df['B365H'] < 1).sum() + (df['B365D'] < 1).sum() + (df['B365A'] < 1).sum()}")
print(f"Missing FTR? {df['FTR'].isna().sum()}")
print(f"Invalid FTR values? {(~df['FTR'].isin(['H', 'D', 'A'])).sum()}")

# 8. Check bookmaker margin
print("\n8. BOOKMAKER MARGIN")
print(f"Average margin: {df['B365_margin'].mean():.2%}")
print(f"Min margin: {df['B365_margin'].min():.2%}")
print(f"Max margin: {df['B365_margin'].max():.2%}")

print("\n" + "=" * 70)