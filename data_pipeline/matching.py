"""
Tennis Era Analysis - Data Matching
Flexible matching strategies for integrating point-by-point data with match-level data.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm

from config.constants import ENHANCED_FUZZY_THRESHOLD, FUZZY_MATCH_THRESHOLD, LLM_MATCH_THRESHOLD, MATCHING_BATCH_SIZE

# Optional OpenAI import
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False


class BaseMatcher(ABC):
    """Base class for all matching strategies."""

    def __init__(self, threshold: float = 80, batch_size: int = MATCHING_BATCH_SIZE):
        self.threshold = threshold
        self.batch_size = batch_size
        self.matches_found = 0
        self.total_processed = 0

    @abstractmethod
    def calculate_match_score(self, pbp_row: pd.Series, match_row: pd.Series) -> float:
        """Calculate similarity score between PBP and match records."""
        pass

    def find_matches(self, pbp_df: pd.DataFrame, match_df: pd.DataFrame) -> pd.DataFrame:
        """Find matches between PBP and match data."""
        print(f"\n🔍 Running {self.__class__.__name__}...")
        print(f"   Threshold: {self.threshold}")
        print(f"   PBP records: {len(pbp_df):,}")
        print(f"   Match records: {len(match_df):,}")

        matches = []
        start_time = time.time()

        # Process in batches
        for i in tqdm(range(0, len(pbp_df), self.batch_size), desc="Matching"):
            batch = pbp_df.iloc[i : i + self.batch_size]
            batch_matches = self._process_batch(batch, match_df)
            matches.extend(batch_matches)
            self.total_processed += len(batch)

        elapsed = time.time() - start_time
        self.matches_found = len(matches)

        print(f"   ✅ Found {self.matches_found:,} matches in {elapsed:.1f}s")
        print(f"   Match rate: {self.matches_found/len(pbp_df)*100:.1f}%")

        return pd.DataFrame(matches)

    def _process_batch(self, pbp_batch: pd.DataFrame, match_df: pd.DataFrame) -> List[Dict]:
        """Process a batch of PBP records."""
        batch_matches = []

        for _, pbp_row in pbp_batch.iterrows():
            best_match = self._find_best_match(pbp_row, match_df)
            if best_match:
                batch_matches.append(best_match)

        return batch_matches

    def _find_best_match(self, pbp_row: pd.Series, match_df: pd.DataFrame) -> Optional[Dict]:
        """Find the best match for a single PBP record."""
        best_score = 0
        best_match = None

        for _, match_row in match_df.iterrows():
            score = self.calculate_match_score(pbp_row, match_row)

            if score > best_score and score >= self.threshold:
                best_score = score
                best_match = {
                    "pbp_id": pbp_row.get("pbp_id"),
                    "match_id": match_row.get("match_id"),
                    "score": score,
                    "strategy": self.__class__.__name__,
                }

        return best_match


class FuzzyMatcher(BaseMatcher):
    """Basic fuzzy string matching strategy."""

    def calculate_match_score(self, pbp_row: pd.Series, match_row: pd.Series) -> float:
        """Calculate fuzzy match score based on player names."""
        # Get player names from both records
        pbp_players = [pbp_row.get("player1", ""), pbp_row.get("player2", "")]
        match_players = [match_row.get("winner_name", ""), match_row.get("loser_name", "")]

        # Remove empty strings
        pbp_players = [p for p in pbp_players if p and str(p) != "nan"]
        match_players = [p for p in match_players if p and str(p) != "nan"]

        if len(pbp_players) < 2 or len(match_players) < 2:
            return 0

        # Calculate all possible name combinations
        scores = []
        for pbp_name in pbp_players:
            for match_name in match_players:
                score = fuzz.ratio(str(pbp_name).lower(), str(match_name).lower())
                scores.append(score)

        # Return average of top 2 scores (both players should match)
        scores.sort(reverse=True)
        return np.mean(scores[:2]) if len(scores) >= 2 else 0


class EnhancedFuzzyMatcher(BaseMatcher):
    """Enhanced fuzzy matching with multiple signals."""

    def calculate_match_score(self, pbp_row: pd.Series, match_row: pd.Series) -> float:
        """Calculate enhanced match score using multiple signals."""
        scores = []
        weights = []

        # 1. Player name matching (weight: 0.6)
        name_score = self._calculate_name_score(pbp_row, match_row)
        if name_score > 0:
            scores.append(name_score)
            weights.append(0.6)

        # 2. Date proximity (weight: 0.2)
        date_score = self._calculate_date_score(pbp_row, match_row)
        if date_score > 0:
            scores.append(date_score)
            weights.append(0.2)

        # 3. Tournament context (weight: 0.2)
        tournament_score = self._calculate_tournament_score(pbp_row, match_row)
        if tournament_score > 0:
            scores.append(tournament_score)
            weights.append(0.2)

        # Calculate weighted average
        if not scores:
            return 0

        weighted_score = np.average(scores, weights=weights[: len(scores)])
        return weighted_score

    def _calculate_name_score(self, pbp_row: pd.Series, match_row: pd.Series) -> float:
        """Calculate name matching score."""
        pbp_players = [pbp_row.get("player1", ""), pbp_row.get("player2", "")]
        match_players = [match_row.get("winner_name", ""), match_row.get("loser_name", "")]

        # Remove empty strings
        pbp_players = [p for p in pbp_players if p and str(p) != "nan"]
        match_players = [p for p in match_players if p and str(p) != "nan"]

        if len(pbp_players) < 2 or len(match_players) < 2:
            return 0

        # Use token sort ratio for better matching
        scores = []
        for pbp_name in pbp_players:
            for match_name in match_players:
                score = fuzz.token_sort_ratio(str(pbp_name).lower(), str(match_name).lower())
                scores.append(score)

        scores.sort(reverse=True)
        return np.mean(scores[:2]) if len(scores) >= 2 else 0

    def _calculate_date_score(self, pbp_row: pd.Series, match_row: pd.Series) -> float:
        """Calculate date proximity score."""
        try:
            pbp_date = pd.to_datetime(pbp_row.get("date"), errors="coerce")
            match_date = pd.to_datetime(match_row.get("tourney_date"), errors="coerce")

            if pd.isna(pbp_date) or pd.isna(match_date):
                return 0

            # Calculate days difference
            days_diff = abs((pbp_date - match_date).days)

            # Score decreases with distance (100 for same day, 0 for >30 days)
            if days_diff == 0:
                return 100
            elif days_diff <= 7:
                return 90
            elif days_diff <= 30:
                return max(0, 100 - days_diff * 3)
            else:
                return 0

        except Exception:
            return 0

    def _calculate_tournament_score(self, pbp_row: pd.Series, match_row: pd.Series) -> float:
        """Calculate tournament context score."""
        # Tournament name matching
        pbp_tournament = str(pbp_row.get("tournament", "")).lower()
        match_tournament = str(match_row.get("tourney_name", "")).lower()

        if not pbp_tournament or not match_tournament:
            return 0

        # Use partial ratio for tournament names
        tournament_score = fuzz.partial_ratio(pbp_tournament, match_tournament)

        # Round matching
        pbp_round = str(pbp_row.get("round", "")).lower()
        match_round = str(match_row.get("round", "")).lower()

        round_score = 0
        if pbp_round and match_round:
            round_score = fuzz.ratio(pbp_round, match_round)

        # Combine tournament and round scores
        return tournament_score * 0.7 + round_score * 0.3


class LLMMatcher(BaseMatcher):
    """LLM-based matching strategy using OpenAI API."""

    def __init__(self, threshold: float = LLM_MATCH_THRESHOLD, batch_size: int = 50, api_key: str = None):
        super().__init__(threshold, batch_size)
        self.api_key = api_key
        if api_key and OPENAI_AVAILABLE:
            openai.api_key = api_key
        elif api_key and not OPENAI_AVAILABLE:
            print("⚠️ OpenAI not available, will use fallback matching")
        self.cache = {}  # Cache LLM responses to save API calls

    def calculate_match_score(self, pbp_row: pd.Series, match_row: pd.Series) -> float:
        """Use LLM to determine match similarity."""

        # Check if OpenAI is available and API key is provided
        if not OPENAI_AVAILABLE or not self.api_key:
            # Fallback to enhanced fuzzy matching
            fallback_matcher = EnhancedFuzzyMatcher(threshold=self.threshold)
            return fallback_matcher.calculate_match_score(pbp_row, match_row)

        # Create cache key
        cache_key = self._create_cache_key(pbp_row, match_row)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Prepare prompt for LLM
        prompt = self._create_matching_prompt(pbp_row, match_row)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a tennis data matching expert. Return only a number between 0-100 indicating match confidence.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=0,
            )

            score_text = response.choices[0].message.content.strip()
            score = float(score_text)

            # Cache the result
            self.cache[cache_key] = score
            return score

        except Exception as e:
            print(f"LLM matching error: {e}")
            # Fallback to enhanced fuzzy matching
            fallback_matcher = EnhancedFuzzyMatcher(threshold=self.threshold)
            score = fallback_matcher.calculate_match_score(pbp_row, match_row)
            self.cache[cache_key] = score
            return score

    def _create_cache_key(self, pbp_row: pd.Series, match_row: pd.Series) -> str:
        """Create a cache key for the LLM request."""
        pbp_key = f"{pbp_row.get('player1', '')}-{pbp_row.get('player2', '')}-{pbp_row.get('date', '')}"
        match_key = f"{match_row.get('winner_name', '')}-{match_row.get('loser_name', '')}-{match_row.get('tourney_date', '')}"
        return f"{pbp_key}|{match_key}"

    def _create_matching_prompt(self, pbp_row: pd.Series, match_row: pd.Series) -> str:
        """Create a prompt for LLM matching."""
        return f"""
        Match these tennis records (0-100 confidence):
        
        PBP Record:
        - Players: {pbp_row.get('player1', 'N/A')} vs {pbp_row.get('player2', 'N/A')}
        - Date: {pbp_row.get('date', 'N/A')}
        - Tournament: {pbp_row.get('tournament', 'N/A')}
        - Round: {pbp_row.get('round', 'N/A')}
        
        Match Record:
        - Players: {match_row.get('winner_name', 'N/A')} vs {match_row.get('loser_name', 'N/A')}
        - Date: {match_row.get('tourney_date', 'N/A')}
        - Tournament: {match_row.get('tourney_name', 'N/A')}
        - Round: {match_row.get('round', 'N/A')}
        
        Confidence (0-100):
        """


def run_matching_experiment(pbp_df: pd.DataFrame, atp_df: pd.DataFrame, openai_api_key: str = None) -> pd.DataFrame:
    """Run matching experiment with different strategies."""

    print("=== MATCHING STRATEGY COMPARISON ===")

    # Define strategies to test
    strategies_config = {
        "fuzzy": {"threshold": FUZZY_MATCH_THRESHOLD},
        "enhanced_fuzzy": {"threshold": ENHANCED_FUZZY_THRESHOLD},
    }

    # Add LLM if API key provided and OpenAI is available
    if openai_api_key and OPENAI_AVAILABLE:
        strategies_config["llm"] = {"threshold": LLM_MATCH_THRESHOLD, "api_key": openai_api_key, "batch_size": 50}
    elif openai_api_key and not OPENAI_AVAILABLE:
        print("⚠️ OpenAI API key provided but OpenAI module not available. Skipping LLM matching.")

    # Run each strategy
    all_results = []
    strategy_performance = {}

    for strategy_name, config in strategies_config.items():
        print(f"\n{'='*50}")
        print(f"STRATEGY: {strategy_name.upper()}")
        print(f"{'='*50}")

        # Create matcher instance
        if strategy_name == "fuzzy":
            matcher = FuzzyMatcher(**config)
        elif strategy_name == "enhanced_fuzzy":
            matcher = EnhancedFuzzyMatcher(**config)
        elif strategy_name == "llm":
            matcher = LLMMatcher(**config)

        # Run matching
        start_time = time.time()
        results = matcher.find_matches(pbp_df, atp_df)
        elapsed = time.time() - start_time

        # Store performance metrics
        strategy_performance[strategy_name] = {
            "matches_found": len(results),
            "match_rate": len(results) / len(pbp_df) * 100,
            "avg_score": results["score"].mean() if len(results) > 0 else 0,
            "processing_time": elapsed,
        }

        # Add strategy column and append to all results
        if len(results) > 0:
            results["strategy"] = strategy_name
            all_results.append(results)

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
    else:
        combined_results = pd.DataFrame()

    # Print summary
    print(f"\n{'='*60}")
    print("MATCHING EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    for strategy, metrics in strategy_performance.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Matches found: {metrics['matches_found']:,}")
        print(f"  Match rate: {metrics['match_rate']:.1f}%")
        print(f"  Avg score: {metrics['avg_score']:.1f}")
        print(f"  Processing time: {metrics['processing_time']:.1f}s")

    return combined_results


if __name__ == "__main__":
    # Example usage
    print("🎾 Tennis Data Matching Module")
    print("This module provides flexible matching strategies for PBP integration.")
    print("Import and use the matchers in your data pipeline.")
