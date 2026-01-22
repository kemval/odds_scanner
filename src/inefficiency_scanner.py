import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class MarketInefficiencyScanner:
    """
    Identifies systematic biases in betting markets
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.inefficiencies = {}
        
    def analyze_home_bias(self) -> Dict:
        """
        Detect if market systematically over/undervalues home teams
        """
        # Calculate expected vs actual home wins
        expected_home_wins = self.df['B365_true_prob_H'].sum()
        actual_home_wins = self.df['Outcome_H'].sum()
        total_matches = len(self.df)
        
        bias = (expected_home_wins - actual_home_wins) / total_matches
        
        # Statistical significance (Chi-square test)
        from scipy.stats import chisquare
        observed = [actual_home_wins, total_matches - actual_home_wins]
        expected = [expected_home_wins, total_matches - expected_home_wins]
        chi2, p_value = chisquare(observed, expected)
        
        result = {
            'bias_type': 'home_bias',
            'expected_wins': expected_home_wins,
            'actual_wins': actual_home_wins,
            'bias_percentage': bias,
            'direction': 'overvalued' if bias > 0 else 'undervalued',
            'statistically_significant': p_value < 0.05,
            'p_value': p_value,
            'sample_size': total_matches
        }
        
        self.inefficiencies['home_bias'] = result
        return result
    
    def analyze_favorite_longshot_bias(self) -> Dict:
        """
        Check if favorites underperform and longshots overperform their odds
        """
        # Define probability buckets
        bins = [0, 0.3, 0.5, 0.7, 1.0]
        labels = ['Longshot', 'Underdog', 'Favorite', 'Heavy_Favorite']
        
        results = []
        
        for outcome_type in ['H', 'D', 'A']:
            prob_col = f'B365_true_prob_{outcome_type}'
            outcome_col = f'Outcome_{outcome_type}'
            
            temp_df = self.df[[prob_col, outcome_col]].dropna().copy()
            temp_df['bucket'] = pd.cut(temp_df[prob_col], bins=bins, labels=labels)
            
            # FIXED: Use proper aggregation
            bucket_analysis = temp_df.groupby('bucket', observed=True).agg(
                Avg_Implied_Prob=(prob_col, 'mean'),
                Count=(prob_col, 'count'),
                Actual_Win_Rate=(outcome_col, 'mean')
            ).reset_index()
            
            bucket_analysis.columns = ['Bucket', 'Avg_Implied_Prob', 'Count', 'Actual_Win_Rate']
            bucket_analysis['Outcome_Type'] = outcome_type
            bucket_analysis['Bias'] = bucket_analysis['Avg_Implied_Prob'] - bucket_analysis['Actual_Win_Rate']
            
            results.append(bucket_analysis)
        
        combined = pd.concat(results, ignore_index=True)
        
        # Find systematic patterns
        heavy_favorites = combined[combined['Bucket'] == 'Heavy_Favorite']
        longshots = combined[combined['Bucket'] == 'Longshot']
        
        result = {
            'bias_type': 'favorite_longshot',
            'full_analysis': combined,
            'heavy_favorite_bias': heavy_favorites['Bias'].mean(),
            'longshot_bias': longshots['Bias'].mean(),
            'interpretation': self._interpret_fl_bias(heavy_favorites['Bias'].mean(), longshots['Bias'].mean())
        }
        
        self.inefficiencies['favorite_longshot'] = result
        return result
    
    def _interpret_fl_bias(self, fav_bias: float, long_bias: float) -> str:
        """Interpret favorite-longshot bias"""
        if fav_bias > 0.02 and long_bias < -0.02:
            return "Classic favorite-longshot bias detected: favorites overvalued, longshots undervalued"
        elif fav_bias < -0.02 and long_bias > 0.02:
            return "Reverse bias: favorites undervalued, longshots overvalued"
        else:
            return "No significant favorite-longshot bias"
    
    def analyze_draw_bias(self) -> Dict:
        """
        Analyze if draw outcomes are systematically mispriced
        """
        expected_draws = self.df['B365_true_prob_D'].sum()
        actual_draws = self.df['Outcome_D'].sum()
        total_matches = len(self.df)
        
        bias = (expected_draws - actual_draws) / total_matches
        
        # By league - FIXED VERSION
        league_analysis = self.df.groupby('League').agg(
            Implied_Draw_Prob=('B365_true_prob_D', 'mean'),
            Actual_Draw_Rate=('Outcome_D', 'mean'),
            Matches=('League', 'count')
        ).reset_index()
        
        league_analysis['Bias'] = league_analysis['Implied_Draw_Prob'] - league_analysis['Actual_Draw_Rate']
        
        result = {
            'bias_type': 'draw_bias',
            'overall_bias': bias,
            'direction': 'overvalued' if bias > 0 else 'undervalued',
            'by_league': league_analysis,
            'strongest_bias_league': league_analysis.loc[league_analysis['Bias'].abs().idxmax(), 'League']
        }
        
        self.inefficiencies['draw_bias'] = result
        return result
    
    def analyze_close_match_bias(self) -> Dict:
        """
        Check if close matches (similar probabilities) are mispriced
        """
        # Filter for close matches (prob difference < 15%)
        close_matches = self.df[self.df['Favorite_Strength'] < 0.15].copy()
        
        if len(close_matches) == 0:
            return {'bias_type': 'close_match', 'error': 'No close matches found'}
        
        # In close matches, implied probabilities should be near 50/50
        # Check if home teams still win more often
        actual_home_rate = close_matches['Outcome_H'].mean()
        implied_home_rate = close_matches['B365_true_prob_H'].mean()
        
        result = {
            'bias_type': 'close_match',
            'sample_size': len(close_matches),
            'implied_home_prob': implied_home_rate,
            'actual_home_rate': actual_home_rate,
            'bias': implied_home_rate - actual_home_rate,
            'interpretation': 'Home advantage persists even in close matches' if actual_home_rate > 0.5 else 'No clear bias'
        }
        
        self.inefficiencies['close_match'] = result
        return result
    
    def run_full_scan(self) -> Dict:
        """
        Run all inefficiency analyses
        """
        print("🔍 Running Market Inefficiency Scan...")
        print("=" * 60)
        
        # Run all analyses
        home = self.analyze_home_bias()
        print(f"\n✓ Home Bias: {home['direction']} by {abs(home['bias_percentage']):.2%}")
        
        fl = self.analyze_favorite_longshot_bias()
        print(f"✓ Favorite-Longshot: {fl['interpretation']}")
        
        draw = self.analyze_draw_bias()
        print(f"✓ Draw Bias: Draws {draw['direction']} by {abs(draw['overall_bias']):.2%}")
        
        close = self.analyze_close_match_bias()
        print(f"✓ Close Match Analysis: {close.get('interpretation', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("✓ Scan complete! Inefficiencies identified.\n")
        
        return self.inefficiencies
    
    def get_actionable_insights(self) -> List[str]:
        """
        Generate betting strategy recommendations based on inefficiencies
        """
        insights = []
        
        if 'home_bias' in self.inefficiencies:
            bias = self.inefficiencies['home_bias']
            if bias['direction'] == 'overvalued' and bias['bias_percentage'] > 0.02:
                insights.append(
                    f"🎯 OPPORTUNITY: Home teams overvalued by {bias['bias_percentage']:.2%}. "
                    "Consider betting on away teams or avoiding home favorites."
                )
            elif bias['direction'] == 'undervalued' and abs(bias['bias_percentage']) > 0.02:
                insights.append(
                    f"🎯 OPPORTUNITY: Home teams undervalued by {abs(bias['bias_percentage']):.2%}. "
                    "Consider betting on home teams, especially in close matches."
                )
        
        if 'favorite_longshot' in self.inefficiencies:
            fl = self.inefficiencies['favorite_longshot']
            if fl['heavy_favorite_bias'] > 0.03:
                insights.append(
                    f"🎯 OPPORTUNITY: Heavy favorites underperform by {fl['heavy_favorite_bias']:.2%}. "
                    "Avoid betting on heavy favorites (>70% implied probability)."
                )
            if fl['longshot_bias'] < -0.03:
                insights.append(
                    f"🎯 OPPORTUNITY: Longshots overperform by {abs(fl['longshot_bias']):.2%}. "
                    "Consider value bets on underdogs (<30% implied probability)."
                )
        
        if 'draw_bias' in self.inefficiencies:
            draw = self.inefficiencies['draw_bias']
            if draw['direction'] == 'undervalued' and abs(draw['overall_bias']) > 0.02:
                insights.append(
                    f"🎯 OPPORTUNITY: Draws undervalued by {abs(draw['overall_bias']):.2%}. "
                    "Consider draw bets, especially in leagues with high draw rates."
                )
            elif draw['direction'] == 'overvalued' and abs(draw['overall_bias']) > 0.02:
                insights.append(
                    f"🎯 OPPORTUNITY: Draws overvalued by {abs(draw['overall_bias']):.2%}. "
                    "Avoid draw bets; focus on home/away outcomes."
                )
        
        if 'close_match' in self.inefficiencies:
            close = self.inefficiencies['close_match']
            if 'bias' in close and abs(close['bias']) > 0.03:
                if close['bias'] > 0:
                    insights.append(
                        f"🎯 OPPORTUNITY: In close matches, home teams overvalued by {close['bias']:.2%}. "
                        "Consider betting away in evenly-matched games."
                    )
        
        if not insights:
            insights.append("📊 No strong exploitable inefficiencies detected above 2% threshold. Market appears well-calibrated.")
        
        return insights


if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv('data/processed/odds_processed.csv')
    
    # Run scanner
    scanner = MarketInefficiencyScanner(df)
    inefficiencies = scanner.run_full_scan()
    
    # Get insights
    print("\n📊 ACTIONABLE INSIGHTS:")
    print("=" * 60)
    for insight in scanner.get_actionable_insights():
        print(f"\n{insight}")
    
    # Save results
    import json
    from pathlib import Path
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    def convert_to_serializable(obj):
        """Convert non-JSON-serializable objects"""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    with open('results/inefficiencies.json', 'w') as f:
        save_dict = convert_to_serializable(inefficiencies)
        json.dump(save_dict, f, indent=2)
    
    print("\n✓ Results saved to results/inefficiencies.json")