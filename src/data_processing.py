import pandas as pd
import numpy as np
from pathlib import Path

class OddsProcessor:
    """
    Process and engineer features from odds data
    """
    
    def __init__(self, df):
        self.df = df.copy()
        
    def clean_data(self):
        """Remove rows with missing critical data"""
        print(f"Original rows: {len(self.df)}")
        
        # Keep only rows with result and at least one bookmaker's odds
        self.df = self.df.dropna(subset=['FTR'])
        self.df = self.df.dropna(subset=['B365H', 'B365D', 'B365A'], how='all')
        
        print(f"After cleaning: {len(self.df)}")
        return self
    
    def odds_to_probability(self, odds):
        """
        Convert decimal odds to implied probability
        Formula: probability = 1 / odds
        """
        return 1 / odds
    
    def calculate_implied_probabilities(self):
        """
        Calculate implied probabilities from odds
        """
        # We'll use Bet365 as our primary bookmaker (most complete data)
        bookmaker = 'B365'
        
        self.df[f'{bookmaker}_prob_H'] = self.odds_to_probability(self.df[f'{bookmaker}H'])
        self.df[f'{bookmaker}_prob_D'] = self.odds_to_probability(self.df[f'{bookmaker}D'])
        self.df[f'{bookmaker}_prob_A'] = self.odds_to_probability(self.df[f'{bookmaker}A'])
        
        # Calculate the bookmaker's margin (overround)
        self.df[f'{bookmaker}_margin'] = (
            self.df[f'{bookmaker}_prob_H'] + 
            self.df[f'{bookmaker}_prob_D'] + 
            self.df[f'{bookmaker}_prob_A']
        ) - 1
        
        # True probabilities (remove margin by normalization)
        total_prob = (self.df[f'{bookmaker}_prob_H'] + 
                     self.df[f'{bookmaker}_prob_D'] + 
                     self.df[f'{bookmaker}_prob_A'])
        
        self.df[f'{bookmaker}_true_prob_H'] = self.df[f'{bookmaker}_prob_H'] / total_prob
        self.df[f'{bookmaker}_true_prob_D'] = self.df[f'{bookmaker}_prob_D'] / total_prob
        self.df[f'{bookmaker}_true_prob_A'] = self.df[f'{bookmaker}_prob_A'] / total_prob
        
        return self
    
    def create_outcome_labels(self):
        """
        Create binary outcome labels for each result type
        """
        self.df['Outcome_H'] = (self.df['FTR'] == 'H').astype(int)
        self.df['Outcome_D'] = (self.df['FTR'] == 'D').astype(int)
        self.df['Outcome_A'] = (self.df['FTR'] == 'A').astype(int)
        
        return self
    
    def identify_favorites(self):
        """
        Identify favorites, underdogs, and close matches
        """
        # Home is favorite if prob_H > prob_A
        self.df['Home_Favorite'] = (
            self.df['B365_true_prob_H'] > self.df['B365_true_prob_A']
        ).astype(int)
        
        # Strength of favorite (probability difference)
        self.df['Favorite_Strength'] = abs(
            self.df['B365_true_prob_H'] - self.df['B365_true_prob_A']
        )
        
        # Classify match closeness
        self.df['Match_Type'] = pd.cut(
            self.df['Favorite_Strength'],
            bins=[0, 0.15, 0.30, 1.0],
            labels=['Close', 'Medium', 'Heavy_Favorite']
        )
        
        return self
    
    def get_processed_data(self):
        """Return processed dataframe"""
        return self.df
    
    def save_processed_data(self, path='data/processed/odds_processed.csv'):
        """Save processed data"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(save_path, index=False)
        print(f"✓ Processed data saved to {save_path}")


if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv('data/raw/combined_raw.csv')
    
    # Process
    processor = OddsProcessor(df)
    processed_df = (processor
                   .clean_data()
                   .calculate_implied_probabilities()
                   .create_outcome_labels()
                   .identify_favorites()
                   .get_processed_data())
    
    # Save
    processor.save_processed_data()
    
    # Quick stats
    print("\n=== Quick Stats ===")
    print(f"Total matches: {len(processed_df)}")
    print(f"\nOutcome distribution:")
    print(processed_df['FTR'].value_counts())
    print(f"\nMatch type distribution:")
    print(processed_df['Match_Type'].value_counts())
    print(f"\nAverage bookmaker margin: {processed_df['B365_margin'].mean():.2%}")