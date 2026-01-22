import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class BettingBacktester:
    """
    Backtest betting strategies based on market inefficiencies
    """
    
    def __init__(self, df: pd.DataFrame, starting_bankroll: float = 1000):
        self.df = df.copy()
        self.starting_bankroll = starting_bankroll
        self.results = {}
        
    def calculate_edge(self, true_prob: float, odds: float) -> float:
        """
        Calculate the edge (expected value) of a bet
        Edge = (true_prob * odds) - 1
        Positive edge = +EV bet
        """
        return (true_prob * odds) - 1
    
    def kelly_criterion(self, edge: float, odds: float) -> float:
        """
        Calculate optimal bet size using Kelly Criterion
        Kelly % = edge / (odds - 1)
        """
        if edge <= 0 or odds <= 1:
            return 0
        return edge / (odds - 1)
    
    def fixed_stake_betting(self, strategy_name: str, bet_selection_func, 
                           stake_pct: float = 0.02) -> Dict:
        """
        FLAT stake betting - stake stays constant (e.g., always $20 per bet)
        This prevents death spiral from losing streaks
        
        Args:
            strategy_name: Name of the strategy
            bet_selection_func: Function that adds bet columns and returns dataframe
            stake_pct: Percentage of STARTING bankroll per bet (default 2%)
        """
        # Get bets with selection applied
        bets = bet_selection_func(self.df.copy())
        
        # Filter to only rows where we should bet
        if 'should_bet' not in bets.columns:
            return {
                'strategy': strategy_name,
                'total_bets': 0,
                'error': 'Strategy function must add should_bet column'
            }
        
        active_bets = bets[bets['should_bet']].copy()
        
        if len(active_bets) == 0:
            return {
                'strategy': strategy_name,
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'starting_bankroll': self.starting_bankroll,
                'final_bankroll': self.starting_bankroll,
                'total_profit': 0,
                'roi': 0,
                'bankroll_history': [self.starting_bankroll],
                'avg_odds': 0,
                'total_staked': 0
            }
        
        # FIXED: Use flat stake based on starting bankroll
        flat_stake = self.starting_bankroll * stake_pct
        
        # Calculate results
        total_profit = 0
        bankroll_history = [self.starting_bankroll]
        
        wins = 0
        losses = 0
        
        for idx, row in active_bets.iterrows():
            # Determine outcome and payout
            if row['bet_outcome'] == row['FTR']:
                profit = flat_stake * (row['bet_odds'] - 1)  # Win
                wins += 1
            else:
                profit = -flat_stake  # Loss
                losses += 1
            
            total_profit += profit
            bankroll_history.append(self.starting_bankroll + total_profit)
        
        final_bankroll = self.starting_bankroll + total_profit
        total_staked = len(active_bets) * flat_stake
        
        result = {
            'strategy': strategy_name,
            'total_bets': len(active_bets),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(active_bets),
            'starting_bankroll': self.starting_bankroll,
            'final_bankroll': final_bankroll,
            'total_profit': total_profit,
            'roi': (total_profit / total_staked) * 100,
            'bankroll_history': bankroll_history,
            'avg_odds': active_bets['bet_odds'].mean(),
            'total_staked': total_staked
        }
        
        return result
    
    def value_betting_strategy(self, min_edge: float = 0.03) -> Dict:
        """
        Strategy 1: Bet when we have positive expected value > threshold
        """
        def select_bets(df):
            # Calculate edges for each outcome
            df['edge_H'] = self.calculate_edge(df['B365_true_prob_H'], df['B365H'])
            df['edge_D'] = self.calculate_edge(df['B365_true_prob_D'], df['B365D'])
            df['edge_A'] = self.calculate_edge(df['B365_true_prob_A'], df['B365A'])
            
            # Find best edge for each match
            df['max_edge'] = df[['edge_H', 'edge_D', 'edge_A']].max(axis=1)
            df['best_bet'] = df[['edge_H', 'edge_D', 'edge_A']].idxmax(axis=1)
            
            # Map to outcome and odds
            df['bet_outcome'] = df['best_bet'].map({'edge_H': 'H', 'edge_D': 'D', 'edge_A': 'A'})
            df['bet_odds'] = df.apply(
                lambda row: row['B365H'] if row['bet_outcome'] == 'H' 
                else (row['B365D'] if row['bet_outcome'] == 'D' else row['B365A']), 
                axis=1
            )
            
            # Only bet when edge exceeds threshold
            df['should_bet'] = df['max_edge'] >= min_edge
            
            return df
        
        return self.fixed_stake_betting(f'Value Betting (edge≥{min_edge:.1%})', select_bets)
    
    def favorite_fade_strategy(self, favorite_threshold: float = 0.65) -> Dict:
        """
        Strategy 2: Bet against heavy favorites (they tend to be overvalued)
        """
        def select_bets(df):
            # Identify heavy favorites
            df['is_home_favorite'] = df['B365_true_prob_H'] >= favorite_threshold
            df['is_away_favorite'] = df['B365_true_prob_A'] >= favorite_threshold
            
            # Initialize bet columns
            df['bet_outcome'] = None
            df['bet_odds'] = None
            df['should_bet'] = False
            
            # Bet against home favorites (bet away)
            home_fav_mask = df['is_home_favorite']
            df.loc[home_fav_mask, 'bet_outcome'] = 'A'
            df.loc[home_fav_mask, 'bet_odds'] = df.loc[home_fav_mask, 'B365A']
            df.loc[home_fav_mask, 'should_bet'] = True
            
            # Bet against away favorites (bet home)
            away_fav_mask = df['is_away_favorite'] & ~home_fav_mask  # Avoid double betting
            df.loc[away_fav_mask, 'bet_outcome'] = 'H'
            df.loc[away_fav_mask, 'bet_odds'] = df.loc[away_fav_mask, 'B365H']
            df.loc[away_fav_mask, 'should_bet'] = True
            
            return df
        
        return self.fixed_stake_betting(f'Fade Favorites (>{favorite_threshold:.0%})', select_bets)
    
    def draw_specialist_strategy(self, league: str = None) -> Dict:
        """
        Strategy 3: Bet on draws when undervalued
        """
        def select_bets(df):
            # Filter by league if specified
            if league:
                league_mask = df['League'] == league
            else:
                league_mask = pd.Series([True] * len(df), index=df.index)
            
            # Calculate historical draw rate
            historical_draw_rate = (df.loc[league_mask, 'FTR'] == 'D').mean()
            
            # Bet on draws when market undervalues them
            df['bet_outcome'] = 'D'
            df['bet_odds'] = df['B365D']
            df['should_bet'] = league_mask & (df['B365_true_prob_D'] < historical_draw_rate - 0.01)
            
            return df
        
        strategy_name = f'Draw Specialist' + (f' ({league})' if league else ' (All)')
        return self.fixed_stake_betting(strategy_name, select_bets)
    
    def home_underdog_strategy(self, max_prob: float = 0.40) -> Dict:
        """
        Strategy 4: Bet on home underdogs
        """
        def select_bets(df):
            # Home team is underdog but not complete longshot
            df['bet_outcome'] = 'H'
            df['bet_odds'] = df['B365H']
            df['should_bet'] = (df['B365_true_prob_H'] >= 0.25) & (df['B365_true_prob_H'] <= max_prob)
            
            return df
        
        return self.fixed_stake_betting(f'Home Underdog (≤{max_prob:.0%})', select_bets)
    
    def close_match_away_strategy(self) -> Dict:
        """
        Strategy 5: In close matches, bet away (if home is overvalued)
        """
        def select_bets(df):
            # Bet away in close matches
            df['bet_outcome'] = 'A'
            df['bet_odds'] = df['B365A']
            df['should_bet'] = (df['Favorite_Strength'] < 0.15) & (df['B365_true_prob_H'] > 0.45)
            
            return df
        
        return self.fixed_stake_betting('Close Match Away', select_bets)
    
    def always_home_strategy(self) -> Dict:
        """
        Baseline Strategy: Always bet on home team
        """
        def select_bets(df):
            df['bet_outcome'] = 'H'
            df['bet_odds'] = df['B365H']
            df['should_bet'] = True  # Bet every match
            return df
        
        return self.fixed_stake_betting('Always Bet Home (Baseline)', select_bets)
    
    def always_away_strategy(self) -> Dict:
        """
        Baseline Strategy: Always bet on away team
        """
        def select_bets(df):
            df['bet_outcome'] = 'A'
            df['bet_odds'] = df['B365A']
            df['should_bet'] = True
            return df
        
        return self.fixed_stake_betting('Always Bet Away (Baseline)', select_bets)
    
    def away_value_strategy(self) -> Dict:
        """
        Strategy 6: Bet on away teams when they have value
        """
        def select_bets(df):
            df['edge_A'] = self.calculate_edge(df['B365_true_prob_A'], df['B365A'])
            df['bet_outcome'] = 'A'
            df['bet_odds'] = df['B365A']
            df['should_bet'] = df['edge_A'] > 0.02
            
            return df
        
        return self.fixed_stake_betting('Away Value Bets', select_bets)
    
    def run_all_strategies(self) -> pd.DataFrame:
        """
        Run all strategies and compare results
        """
        print("🎲 Running Backtesting Strategies...")
        print("=" * 60)
        
        strategies = []
        
        # BASELINE STRATEGIES (to verify calculations)
        strategies.append(self.always_home_strategy())
        strategies.append(self.always_away_strategy())
        
        # Core strategies
        strategies.append(self.value_betting_strategy(min_edge=0.03))
        strategies.append(self.value_betting_strategy(min_edge=0.05))
        strategies.append(self.favorite_fade_strategy(favorite_threshold=0.65))
        strategies.append(self.favorite_fade_strategy(favorite_threshold=0.70))
        strategies.append(self.draw_specialist_strategy())
        strategies.append(self.home_underdog_strategy(max_prob=0.40))
        strategies.append(self.home_underdog_strategy(max_prob=0.35))
        strategies.append(self.close_match_away_strategy())
        strategies.append(self.away_value_strategy())
        
        # League-specific draw strategies
        for league in self.df['League'].unique():
            strategies.append(self.draw_specialist_strategy(league=league))
        
        # Filter out strategies with errors or no bets
        valid_strategies = [s for s in strategies if 'error' not in s and s['total_bets'] > 0]
        
        if len(valid_strategies) == 0:
            print("⚠️  No valid strategies with bets found")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(valid_strategies)
        
        # Sort by ROI
        results_df = results_df.sort_values('roi', ascending=False)
        
        # Print results
        print("\n📊 STRATEGY PERFORMANCE SUMMARY")
        print("=" * 60)
        print(results_df[['strategy', 'total_bets', 'win_rate', 'roi', 'total_profit']].to_string(index=False))
        
        self.results = results_df
        return results_df
    
    def plot_strategy_comparison(self, save_path='results/backtest_results.png'):
        """
        Visualize strategy performance
        """
        if len(self.results) == 0:
            print("No results to plot. Run strategies first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. ROI Comparison
        ax1 = axes[0, 0]
        top_strategies = self.results.nlargest(min(10, len(self.results)), 'roi')
        colors = ['green' if x > 0 else 'red' for x in top_strategies['roi']]
        bars = ax1.barh(range(len(top_strategies)), top_strategies['roi'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(top_strategies)))
        ax1.set_yticklabels(top_strategies['strategy'], fontsize=9)
        ax1.set_xlabel('ROI (%)')
        ax1.set_title('Top Strategies by ROI', fontweight='bold', fontsize=12)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(top_strategies.iterrows()):
            ax1.text(row['roi'], i, f" {row['roi']:.1f}%", va='center', fontweight='bold', fontsize=8)
        
        # 2. Win Rate vs ROI
        ax2 = axes[0, 1]
        scatter = ax2.scatter(self.results['win_rate'] * 100, self.results['roi'], 
                             s=self.results['total_bets']/2, alpha=0.6, c=self.results['roi'], 
                             cmap='RdYlGn', edgecolors='black', linewidth=1)
        ax2.set_xlabel('Win Rate (%)')
        ax2.set_ylabel('ROI (%)')
        ax2.set_title('Win Rate vs ROI (size = # of bets)', fontweight='bold', fontsize=12)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='ROI (%)')
        
        # 3. Number of Bets
        ax3 = axes[1, 0]
        top_by_bets = self.results.nlargest(min(10, len(self.results)), 'total_bets')
        ax3.bar(range(len(top_by_bets)), top_by_bets['total_bets'], 
               color='skyblue', alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(top_by_bets)))
        ax3.set_xticklabels(top_by_bets['strategy'], rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Number of Bets')
        ax3.set_title('Strategies by Activity', fontweight='bold', fontsize=12)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Profit Distribution
        ax4 = axes[1, 1]
        profitable = self.results[self.results['roi'] > 0]
        unprofitable = self.results[self.results['roi'] <= 0]
        
        if len(profitable) > 0 and len(unprofitable) > 0:
            data_to_plot = [profitable['roi'].values, unprofitable['roi'].values]
            labels = [f'Profitable\n({len(profitable)})', f'Unprofitable\n({len(unprofitable)})']
            bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True, meanline=True)
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightcoral')
        else:
            ax4.hist(self.results['roi'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('ROI (%)')
        
        ax4.set_ylabel('ROI (%)')
        ax4.set_title('ROI Distribution', fontweight='bold', fontsize=12)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Betting Strategy Backtest Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ Backtest visualization saved to {save_path}")
    
    def plot_bankroll_evolution(self, top_n=5, save_path='results/bankroll_evolution.png'):
        """
        Plot how bankroll evolves over time for top strategies
        """
        if len(self.results) == 0:
            print("No results to plot. Run strategies first.")
            return
        
        top_strategies = self.results.nlargest(min(top_n, len(self.results)), 'roi')
        
        plt.figure(figsize=(14, 8))
        
        for idx, row in top_strategies.iterrows():
            plt.plot(row['bankroll_history'], label=f"{row['strategy']} ({row['roi']:.1f}%)", 
                    linewidth=2, alpha=0.8)
        
        plt.axhline(y=self.starting_bankroll, color='red', linestyle='--', 
                   label=f'Starting (${self.starting_bankroll})', linewidth=2)
        plt.xlabel('Bet Number', fontsize=12)
        plt.ylabel('Bankroll ($)', fontsize=12)
        plt.title(f'Bankroll Evolution - Top {len(top_strategies)} Strategies', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Bankroll evolution chart saved to {save_path}")


if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv('data/processed/odds_processed.csv')
    
    # Initialize backtester
    backtester = BettingBacktester(df, starting_bankroll=1000)
    
    # Run all strategies
    results = backtester.run_all_strategies()
    
    if len(results) > 0:
        # Visualize results
        backtester.plot_strategy_comparison()
        backtester.plot_bankroll_evolution(top_n=5)
        
        # Save results
        results.to_csv('results/backtest_results.csv', index=False)
        print("\n✓ Results saved to results/backtest_results.csv")
        
        # Print best strategy details
        best = results.iloc[0]
        print("\n" + "=" * 60)
        print("🏆 BEST PERFORMING STRATEGY")
        print("=" * 60)
        print(f"Strategy: {best['strategy']}")
        print(f"Total Bets: {best['total_bets']}")
        print(f"Win Rate: {best['win_rate']:.2%}")
        print(f"ROI: {best['roi']:.2f}%")
        print(f"Final Bankroll: ${best['final_bankroll']:.2f}")
        print(f"Total Profit: ${best['total_profit']:.2f}")
    else:
        print("\n⚠️  No strategies generated bets. Check your data and thresholds.")