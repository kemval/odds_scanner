import pandas as pd
import requests
from pathlib import Path
import time

class OddsDataCollector:
    """
    Collects historical soccer match data with odds from football-data.co.uk
    """
    
    def __init__(self, save_dir='data/raw'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://www.football-data.co.uk/mmz4281"
        
    def download_season_data(self, league, season):
        """
        Download data for a specific league and season
        
        Args:
            league: League code (e.g., 'E0' for Premier League)
            season: Season in format 'YYYY' (e.g., '2324' for 2023/24)
        """
        url = f"{self.base_url}/{season}/{league}.csv"
        
        try:
            print(f"Downloading {league} {season}...")
            df = pd.read_csv(url, encoding='latin1')
            
            # Save raw data
            filename = self.save_dir / f"{league}_{season}.csv"
            df.to_csv(filename, index=False)
            print(f"✓ Saved to {filename}")
            
            time.sleep(1)  # Be polite to the server
            return df
            
        except Exception as e:
            print(f"✗ Error downloading {league} {season}: {e}")
            return None
    
    def download_multiple_seasons(self, leagues, seasons):
        """
        Download data for multiple leagues and seasons
        
        Args:
            leagues: List of league codes ['E0', 'E1', 'SP1', etc.]
            seasons: List of seasons ['2122', '2223', '2324']
        """
        all_data = []
        
        for league in leagues:
            for season in seasons:
                df = self.download_season_data(league, season)
                if df is not None:
                    df['League'] = league
                    df['Season'] = season
                    all_data.append(df)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined.to_csv(self.save_dir / 'combined_raw.csv', index=False)
            print(f"\n✓ Combined dataset saved: {len(combined)} matches")
            return combined
        
        return None


# League codes:
# E0 = Premier League, E1 = Championship
# SP1 = La Liga, I1 = Serie A, D1 = Bundesliga
# F1 = Ligue 1

if __name__ == "__main__":
    collector = OddsDataCollector()
    
    # Download last 3 seasons of major leagues
    leagues = ['E0', 'SP1', 'I1', 'D1']  # Top 4 European leagues
    seasons = ['2122', '2223', '2324']   # 2021-2024
    
    data = collector.download_multiple_seasons(leagues, seasons)
    print("\n✓ Data collection complete!")