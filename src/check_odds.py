import pandas as pd

df = pd.read_csv('data/processed/odds_processed.csv')

print("DETAILED ODDS CHECK")
print("=" * 70)

# Check a few specific matches manually
for i in range(5):
    row = df.iloc[i]
    
    print(f"\nMatch {i+1}: {row['HomeTeam']} vs {row['AwayTeam']}")
    print(f"Result: {row['FTR']}")
    print(f"Home odds (B365H): {row['B365H']}")
    
    # If we bet $20 on home
    stake = 20
    if row['FTR'] == 'H':
        # Home wins - we get stake × odds back
        payout = stake * row['B365H']
        profit = payout - stake
        print(f"Outcome: HOME WIN")
        print(f"Payout: ${payout:.2f} (stake × {row['B365H']:.2f})")
        print(f"Profit: ${profit:.2f}")
    else:
        profit = -stake
        print(f"Outcome: HOME LOSS")
        print(f"Profit: $-{stake:.2f}")

# Now let's recalculate CORRECTLY
print("\n" + "=" * 70)
print("RECALCULATING WITH CORRECT FORMULA")
print("=" * 70)

total_profit = 0
stake = 20

for idx, row in df.iterrows():
    if row['FTR'] == 'H':
        # WIN: profit = stake × (odds - 1)
        profit = stake * (row['B365H'] - 1)
    else:
        # LOSS: lose the stake
        profit = -stake
    
    total_profit += profit

total_staked = len(df) * stake
roi = (total_profit / total_staked) * 100

print(f"Total matches: {len(df)}")
print(f"Total staked: ${total_staked:,.2f}")
print(f"Total profit: ${total_profit:,.2f}")
print(f"ROI: {roi:.2f}%")
print(f"Final bankroll (starting $1000): ${1000 + total_profit:,.2f}")

# Compare with original calculation
print("\n" + "=" * 70)
print("COMPARISON:")
print("=" * 70)
print(f"Your verification script profit: $-4,792.40")
print(f"Correct calculation profit: ${total_profit:,.2f}")
print(f"Difference: This explains the discrepancy!")