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

from config.constants import (
    EMBEDDING_MATCH_THRESHOLD,
    EMBEDDING_MODEL,
    FUZZY_MATCH_THRESHOLD,
    MATCHING_BATCH_SIZE,
)
from data_pipeline.caching import EmbeddingCache, MatchingResultsCache


class BaseMatcher(ABC):
    """Base class for all matching strategies."""

    def __init__(
        self,
        threshold: float = 80,
        batch_size: int = MATCHING_BATCH_SIZE,
        date_window_days: int = 30,
        use_tournament_normalization: bool = False,
    ):
        self.threshold = threshold
        self.batch_size = batch_size
        self.date_window_days = date_window_days
        self.use_tournament_normalization = use_tournament_normalization
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

        # Process in batches with better progress tracking
        total_batches = (len(pbp_df) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(pbp_df), self.batch_size), desc="Matching", total=total_batches):
            batch = pbp_df.iloc[i : i + self.batch_size]
            batch_start = time.time()
            batch_matches = self._process_batch(batch, match_df)
            batch_time = time.time() - batch_start

            matches.extend(batch_matches)
            self.total_processed += len(batch)

            # Progress update every 10 batches or if batch takes too long
            if (i // self.batch_size) % 10 == 0 or batch_time > 30:
                elapsed = time.time() - start_time
                rate = self.total_processed / elapsed if elapsed > 0 else 0
                eta = (len(pbp_df) - self.total_processed) / rate if rate > 0 else 0
                print(
                    f"   Progress: {self.total_processed:,}/{len(pbp_df):,} records, "
                    f"{len(matches):,} matches found, ETA: {eta/60:.1f}min"
                )

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
        """Find the best match for a single PBP record with optimized filtering."""
        best_score = 0
        best_match = None

        # Pre-filter matches by date using configured window
        filtered_matches = self._pre_filter_matches(pbp_row, match_df, self.date_window_days)

        # If no matches in date window, fall back to full dataset but with smaller sample
        if filtered_matches.empty:
            # Sample a subset for efficiency if dataset is large
            if len(match_df) > 5000:
                filtered_matches = match_df.sample(n=min(5000, len(match_df)), random_state=42)
            else:
                filtered_matches = match_df

        for _, match_row in filtered_matches.iterrows():
            score = self.calculate_match_score(pbp_row, match_row)

            if score > best_score and score >= self.threshold:
                best_score = score

                # Create match_id if it doesn't exist
                match_id = match_row.get("match_id")
                if match_id is None or pd.isna(match_id):
                    # Create a meaningful ID from available data
                    tourney_id = match_row.get("tourney_id", "unknown")
                    match_num = match_row.get("match_num", match_row.name if hasattr(match_row, "name") else 0)
                    match_id = f"{tourney_id}-{match_num}"

                best_match = {
                    "pbp_id": pbp_row.get("pbp_id"),
                    "match_id": match_id,
                    "score": score,
                    "strategy": self.__class__.__name__,
                }

                # Early termination for very high confidence matches
                if best_score >= 95:
                    break

        return best_match

    def _pre_filter_matches(self, pbp_row: pd.Series, match_df: pd.DataFrame, date_window_days: int = 30) -> pd.DataFrame:
        """Pre-filter match records to reduce search space with optional tournament filtering."""
        try:
            # Fix: Use correct column name for standardized PBP data
            pbp_date = pd.to_datetime(pbp_row.get("date_standardized"), errors="coerce")
            if pd.isna(pbp_date):
                return pd.DataFrame()  # Return empty if no valid date

            # Convert match dates if not already datetime
            if not pd.api.types.is_datetime64_any_dtype(match_df.get("tourney_date", pd.Series())):
                match_dates = pd.to_datetime(match_df.get("tourney_date", pd.Series()), errors="coerce")
            else:
                match_dates = match_df.get("tourney_date", pd.Series())

            # Filter to configurable date window
            date_window = pd.Timedelta(days=date_window_days)
            date_mask = (match_dates >= pbp_date - date_window) & (match_dates <= pbp_date + date_window)

            date_filtered = match_df[date_mask]

            # Additional tournament filtering if enabled and tournament normalization is available
            if (
                self.use_tournament_normalization
                and not date_filtered.empty
                and "tny_name_normalized" in pbp_row.index
                and "tourney_name_normalized" in date_filtered.columns
            ):
                pbp_tournament = pbp_row.get("tny_name_normalized", "").strip().lower()
                if pbp_tournament and pbp_tournament != "nan":
                    # Try exact tournament match first
                    tournament_mask = date_filtered["tourney_name_normalized"].str.lower().str.strip() == pbp_tournament
                    tournament_filtered = date_filtered[tournament_mask]

                    if not tournament_filtered.empty:
                        # Found exact tournament matches within date window
                        return tournament_filtered

                    # If no exact tournament match, expand date window for this tournament
                    expanded_window = pd.Timedelta(days=date_window_days * 2)  # Double the window for same tournament
                    expanded_date_mask = (match_dates >= pbp_date - expanded_window) & (match_dates <= pbp_date + expanded_window)
                    expanded_date_filtered = match_df[expanded_date_mask]

                    if not expanded_date_filtered.empty and "tourney_name_normalized" in expanded_date_filtered.columns:
                        tournament_mask = expanded_date_filtered["tourney_name_normalized"].str.lower().str.strip() == pbp_tournament
                        tournament_filtered = expanded_date_filtered[tournament_mask]

                        if not tournament_filtered.empty:
                            return tournament_filtered

            return date_filtered

        except Exception:
            return pd.DataFrame()  # Return empty on any error


class FuzzyMatcher(BaseMatcher):
    """Basic fuzzy string matching strategy with order-invariant matching."""

    def __init__(
        self,
        threshold: float = FUZZY_MATCH_THRESHOLD,
        batch_size: int = MATCHING_BATCH_SIZE,
        date_window_days: int = 30,
        use_tournament_normalization: bool = False,
    ):
        super().__init__(threshold, batch_size, date_window_days, use_tournament_normalization)

    def _create_player_key(self, player1: str, player2: str) -> frozenset:
        """Create order-invariant key from two player names."""
        # Normalize names (lowercase, strip whitespace)
        p1 = str(player1).strip().lower() if player1 and str(player1) != "nan" else ""
        p2 = str(player2).strip().lower() if player2 and str(player2) != "nan" else ""

        # Only create key if both players are valid
        if p1 and p2:
            return frozenset([p1, p2])
        return frozenset()

    def _find_best_match(self, pbp_row: pd.Series, match_df: pd.DataFrame) -> Optional[Dict]:
        """Find the best match using order-invariant player key filtering."""
        best_score = 0
        best_match = None

        # Create order-invariant key for PBP record
        pbp_key = self._create_player_key(pbp_row.get("server1", ""), pbp_row.get("server2", ""))

        if not pbp_key:
            return None  # Invalid player data

        # Pre-filter matches by date using configured window
        filtered_matches = self._pre_filter_matches(pbp_row, match_df, self.date_window_days)

        # If no matches in date window, fall back to full dataset but with smaller sample
        if filtered_matches.empty:
            # Sample a subset for efficiency if dataset is large
            if len(match_df) > 5000:
                filtered_matches = match_df.sample(n=min(5000, len(match_df)), random_state=42)
            else:
                filtered_matches = match_df

        # First pass: Find exact player key matches for order-invariant matching
        exact_key_matches = []
        for _, match_row in filtered_matches.iterrows():
            match_key = self._create_player_key(match_row.get("winner_name", ""), match_row.get("loser_name", ""))

            if match_key == pbp_key:  # Order-invariant match
                exact_key_matches.append(match_row)

        # Process exact key matches first (highest priority)
        candidates = exact_key_matches if exact_key_matches else filtered_matches.iterrows()

        # Convert to iterable if needed
        if not hasattr(candidates, "__iter__") or hasattr(candidates, "iterrows"):
            candidates = [(_, row) for _, row in candidates]
        elif exact_key_matches:
            candidates = [(None, row) for row in exact_key_matches]

        for _, match_row in candidates:
            score = self.calculate_match_score(pbp_row, match_row)

            if score > best_score and score >= self.threshold:
                best_score = score

                # Create match_id if it doesn't exist
                match_id = match_row.get("match_id")
                if match_id is None or pd.isna(match_id):
                    # Create a meaningful ID from available data
                    tourney_id = match_row.get("tourney_id", "unknown")
                    match_num = match_row.get("match_num", match_row.name if hasattr(match_row, "name") else 0)
                    match_id = f"{tourney_id}-{match_num}"

                best_match = {
                    "pbp_id": pbp_row.get("pbp_id"),
                    "match_id": match_id,
                    "score": score,
                    "strategy": self.__class__.__name__,
                    "order_invariant": True if exact_key_matches else False,
                }

                # Early termination for very high confidence matches from exact key matches
                if best_score >= 95 and exact_key_matches:
                    break

        return best_match

    def calculate_match_score(self, pbp_row: pd.Series, match_row: pd.Series) -> float:
        """Calculate fuzzy match score based on player names."""
        # Get player names from both records (using correct column names)
        pbp_players = [pbp_row.get("server1", ""), pbp_row.get("server2", "")]
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


class EmbeddingMatcher(BaseMatcher):
    """Optimized embedding-based matching strategy with preprocessing and caching."""

    def __init__(
        self,
        threshold: float = EMBEDDING_MATCH_THRESHOLD,
        batch_size: int = 50,
        model: str = EMBEDDING_MODEL,
        date_window_days: int = 30,
        use_tournament_normalization: bool = False,
    ):
        super().__init__(threshold, batch_size, date_window_days, use_tournament_normalization)
        self.model = model

        # Initialize caches
        self.embedding_cache = EmbeddingCache(model=model)

        # Pre-computed embeddings storage
        self.pbp_embeddings = {}
        self.match_embeddings = {}
        self.preprocessed = False

        # Test API availability
        try:
            self.embedding_cache._get_single_embedding("test")
            self._use_fallback = False
            print(f"✅ Embedding API available with model: {self.model}")
        except Exception as e:
            print(f"⚠️ Embedding API not available: {e}")
            print("   Falling back to fuzzy matching")
            self._use_fallback = True

    def _create_player_key(self, player1: str, player2: str) -> frozenset:
        """Create order-invariant key from two player names."""
        # Normalize names (lowercase, strip whitespace)
        p1 = str(player1).strip().lower() if player1 and str(player1) != "nan" else ""
        p2 = str(player2).strip().lower() if player2 and str(player2) != "nan" else ""

        # Only create key if both players are valid
        if p1 and p2:
            return frozenset([p1, p2])
        return frozenset()

    def _find_best_match(self, pbp_row: pd.Series, match_df: pd.DataFrame) -> Optional[Dict]:
        """Find the best match using order-invariant player key filtering."""
        best_score = 0
        best_match = None

        # Create order-invariant key for PBP record
        pbp_key = self._create_player_key(pbp_row.get("server1", ""), pbp_row.get("server2", ""))

        if not pbp_key:
            return None  # Invalid player data

        # Pre-filter matches by date using configured window
        filtered_matches = self._pre_filter_matches(pbp_row, match_df, self.date_window_days)

        # If no matches in date window, fall back to full dataset but with smaller sample
        if filtered_matches.empty:
            # Sample a subset for efficiency if dataset is large
            if len(match_df) > 5000:
                filtered_matches = match_df.sample(n=min(5000, len(match_df)), random_state=42)
            else:
                filtered_matches = match_df

        # First pass: Find exact player key matches for order-invariant matching
        exact_key_matches = []
        for _, match_row in filtered_matches.iterrows():
            match_key = self._create_player_key(match_row.get("winner_name", ""), match_row.get("loser_name", ""))

            if match_key == pbp_key:  # Order-invariant match
                exact_key_matches.append(match_row)

        # Process exact key matches first (highest priority)
        candidates = exact_key_matches if exact_key_matches else filtered_matches.iterrows()

        # Convert to iterable if needed
        if not hasattr(candidates, "__iter__") or hasattr(candidates, "iterrows"):
            candidates = [(_, row) for _, row in candidates]
        elif exact_key_matches:
            candidates = [(None, row) for row in exact_key_matches]

        for _, match_row in candidates:
            score = self.calculate_match_score(pbp_row, match_row)

            if score > best_score and score >= self.threshold:
                best_score = score

                # Create match_id if it doesn't exist
                match_id = match_row.get("match_id")
                if match_id is None or pd.isna(match_id):
                    # Create a meaningful ID from available data
                    tourney_id = match_row.get("tourney_id", "unknown")
                    match_num = match_row.get("match_num", match_row.name if hasattr(match_row, "name") else 0)
                    match_id = f"{tourney_id}-{match_num}"

                best_match = {
                    "pbp_id": pbp_row.get("pbp_id"),
                    "match_id": match_id,
                    "score": score,
                    "strategy": self.__class__.__name__,
                    "order_invariant": True if exact_key_matches else False,
                }

                # Early termination for very high confidence matches from exact key matches
                if best_score >= 95 and exact_key_matches:
                    break

        return best_match

    def preprocess_embeddings(self, pbp_df: pd.DataFrame, match_df: pd.DataFrame):
        """Pre-compute all embeddings for faster matching."""
        if self._use_fallback:
            return

        print("\n🔄 Preprocessing embeddings...")

        # Extract all unique player names
        pbp_names = self._extract_unique_names(pbp_df, ["server1", "server2"])
        match_names = self._extract_unique_names(match_df, ["winner_name", "loser_name"])

        all_names = list(set(pbp_names + match_names))
        print(f"   Found {len(all_names):,} unique player names")

        # Get embeddings for all names (uses caching)
        all_embeddings = self.embedding_cache.get_embeddings(all_names)

        # Store embeddings by record ID for fast lookup
        self._store_record_embeddings(pbp_df, all_embeddings, ["server1", "server2"], "pbp_id")
        self._store_record_embeddings(match_df, all_embeddings, ["winner_name", "loser_name"], "match_id")

        self.preprocessed = True
        print("✅ Preprocessing complete")

    def _extract_unique_names(self, df: pd.DataFrame, name_columns: List[str]) -> List[str]:
        """Extract unique player names from specified columns."""
        names = []
        for col in name_columns:
            if col in df.columns:
                col_names = df[col].dropna().astype(str)
                col_names = col_names[col_names != "nan"].tolist()
                names.extend(col_names)
        return list(set(names))

    def _store_record_embeddings(self, df: pd.DataFrame, all_embeddings: Dict[str, np.ndarray], name_columns: List[str], id_column: str):
        """Store embeddings by record ID for fast lookup during matching."""
        embeddings_dict = self.pbp_embeddings if id_column == "pbp_id" else self.match_embeddings

        for idx, row in df.iterrows():
            # Use the specified ID column if it exists and has a value, otherwise use DataFrame index
            record_id = row.get(id_column)
            if record_id is None or pd.isna(record_id):
                record_id = idx  # Use DataFrame index as fallback

            # Get embeddings for this record's players
            player_embeddings = []
            for col in name_columns:
                name = str(row.get(col, "")).strip().lower()
                if name and name != "nan":
                    embedding = all_embeddings.get(name, np.zeros(384))
                    player_embeddings.append(embedding)

            # Only store if we have at least one valid embedding
            if player_embeddings:
                embeddings_dict[record_id] = player_embeddings

    def calculate_match_score(self, pbp_row: pd.Series, match_row: pd.Series) -> float:
        """Calculate match score using pre-computed embeddings and vectorized operations."""

        if self._use_fallback:
            return self._fallback_matching(pbp_row, match_row)

        try:
            if self.preprocessed:
                return self._calculate_score_preprocessed(pbp_row, match_row)
            else:
                return self._calculate_score_on_demand(pbp_row, match_row)

        except Exception as e:
            print(f"Embedding matching error: {e}")
            return self._fallback_matching(pbp_row, match_row)

    def _calculate_score_preprocessed(self, pbp_row: pd.Series, match_row: pd.Series) -> float:
        """Calculate score using pre-computed embeddings (much faster)."""
        pbp_id = pbp_row.get("pbp_id")
        match_id = match_row.get("match_id")

        # If match_id doesn't exist, use the row's name (index) from the DataFrame
        if match_id is None or pd.isna(match_id):
            match_id = match_row.name if hasattr(match_row, "name") and match_row.name is not None else 0

        pbp_embs = self.pbp_embeddings.get(pbp_id, [])
        match_embs = self.match_embeddings.get(match_id, [])

        # If preprocessing failed for this record pair, fall back to on-demand
        if len(pbp_embs) < 2 or len(match_embs) < 2:
            return self._calculate_score_on_demand(pbp_row, match_row)

        return self._calculate_similarity_matrix(pbp_embs, match_embs)

    def _calculate_score_on_demand(self, pbp_row: pd.Series, match_row: pd.Series) -> float:
        """Calculate score with on-demand embedding generation (fallback)."""
        # Get player names from both records
        pbp_players = [pbp_row.get("server1", ""), pbp_row.get("server2", "")]
        match_players = [match_row.get("winner_name", ""), match_row.get("loser_name", "")]

        # Remove empty strings
        pbp_players = [p for p in pbp_players if p and str(p) != "nan"]
        match_players = [p for p in match_players if p and str(p) != "nan"]

        if len(pbp_players) < 2 or len(match_players) < 2:
            return 0

        # Get embeddings
        pbp_embeddings = self.embedding_cache.get_embeddings(pbp_players)
        match_embeddings = self.embedding_cache.get_embeddings(match_players)

        pbp_embs = [pbp_embeddings[name] for name in pbp_players]
        match_embs = [match_embeddings[name] for name in match_players]

        return self._calculate_similarity_matrix(pbp_embs, match_embs)

    def _calculate_similarity_matrix(self, pbp_embs: List[np.ndarray], match_embs: List[np.ndarray]) -> float:
        """Calculate similarity using vectorized operations."""
        if not pbp_embs or not match_embs:
            return 0

        # Convert to numpy arrays for vectorized operations
        pbp_matrix = np.array(pbp_embs)  # Shape: (n_pbp_players, embedding_dim)
        match_matrix = np.array(match_embs)  # Shape: (n_match_players, embedding_dim)

        # Normalize vectors
        pbp_norms = np.linalg.norm(pbp_matrix, axis=1, keepdims=True)
        match_norms = np.linalg.norm(match_matrix, axis=1, keepdims=True)

        # Avoid division by zero
        pbp_norms = np.where(pbp_norms == 0, 1, pbp_norms)
        match_norms = np.where(match_norms == 0, 1, match_norms)

        pbp_normalized = pbp_matrix / pbp_norms
        match_normalized = match_matrix / match_norms

        # Calculate cosine similarity matrix
        similarity_matrix = np.dot(pbp_normalized, match_normalized.T)

        # Convert to percentage and get top similarities
        similarity_matrix *= 100

        # Get best matches (top 2 similarities for player pairs)
        flat_similarities = similarity_matrix.flatten()
        top_similarities = np.sort(flat_similarities)[-2:]  # Top 2 matches

        return float(np.mean(top_similarities))

    def find_matches(self, pbp_df: pd.DataFrame, match_df: pd.DataFrame) -> pd.DataFrame:
        """Override to add preprocessing step."""
        # Preprocess embeddings for speed
        self.preprocess_embeddings(pbp_df, match_df)

        # Call parent method for actual matching
        return super().find_matches(pbp_df, match_df)

    def _fallback_matching(self, pbp_row: pd.Series, match_row: pd.Series) -> float:
        """Fallback to fuzzy matching."""
        fallback_matcher = FuzzyMatcher(threshold=self.threshold)
        return fallback_matcher.calculate_match_score(pbp_row, match_row)

    def get_cache_stats(self) -> Dict:
        """Get embedding cache statistics."""
        return {
            "cache_size": len(self.embedding_cache.embeddings),
            "preprocessed": self.preprocessed,
            "pbp_records_cached": len(self.pbp_embeddings),
            "match_records_cached": len(self.match_embeddings),
            "cache_file": str(self.embedding_cache.embeddings_file),
        }


def run_matching_experiment(
    pbp_df: pd.DataFrame,
    atp_df: pd.DataFrame,
    strategies: List[str] = None,
    use_cache: bool = True,
    date_window_days: int = 3,
    use_tournament_normalization: bool = False,
) -> pd.DataFrame:
    """
    Run matching experiment with different strategies, with optional result caching.

    Args:
        pbp_df: Point-by-point data
        atp_df: Match-level data
        strategies: List of strategies to test. If None, tests all available.
        use_cache: Whether to use cached results if available
        date_window_days: Date window for fuzzy matching (±days)
        use_tournament_normalization: Whether to use tournament normalization for better matching
    """

    print("=== MATCHING STRATEGY COMPARISON ===")
    print(f"📊 Date window: ±{date_window_days} days (optimized from analysis)")
    if use_tournament_normalization:
        print("🏆 Tournament normalization: ENABLED")
    else:
        print("🏆 Tournament normalization: DISABLED")

    # Initialize result cache
    results_cache = MatchingResultsCache() if use_cache else None

    # Define available strategies
    available_strategies = {
        "fuzzy": {"threshold": FUZZY_MATCH_THRESHOLD},
        "embedding": {"threshold": EMBEDDING_MATCH_THRESHOLD},
    }

    # Filter strategies if specific ones requested
    if strategies:
        available_strategies = {k: v for k, v in available_strategies.items() if k in strategies}

    if not available_strategies:
        print("❌ No valid strategies to test!")
        return pd.DataFrame()

    print(f"📋 Testing {len(available_strategies)} strategies: {list(available_strategies.keys())}")
    if use_cache:
        print("💾 Result caching enabled")

    # Run each strategy
    all_results = []
    strategy_performance = {}

    for strategy_name, config in available_strategies.items():
        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy_name.upper()}")
        print(f"{'='*60}")

        try:
            threshold = config["threshold"]

            # Check for cached results first
            if results_cache and results_cache.has_cached_results(strategy_name, pbp_df, atp_df, threshold):
                print("📂 Loading cached results...")
                results = results_cache.load_cached_results(strategy_name, pbp_df, atp_df, threshold)

                if results is not None and len(results) > 0:
                    # Calculate performance metrics from cached results
                    processing_time = 0  # Cached results are instantaneous
                    strategy_performance[strategy_name] = {
                        "matches_found": len(results),
                        "match_rate": len(results) / len(pbp_df) * 100 if len(pbp_df) > 0 else 0,
                        "avg_score": results["score"].mean() if len(results) > 0 else 0,
                        "processing_time": processing_time,
                        "cached": True,
                    }

                    # Add strategy column and append to all results
                    results["strategy"] = strategy_name
                    all_results.append(results)
                    print(f"✅ {strategy_name} loaded from cache")
                    continue

            # No cached results, run the strategy
            print("🔄 Running strategy...")

            # Create matcher instance
            matcher = _create_matcher(strategy_name, config, date_window_days, use_tournament_normalization)

            if not matcher:
                print(f"❌ Failed to create matcher for {strategy_name}")
                continue

            # Run matching
            start_time = time.time()
            results = matcher.find_matches(pbp_df, atp_df)
            elapsed = time.time() - start_time

            # Cache the results for future use
            if results_cache and len(results) > 0:
                results_cache.save_results(results, strategy_name, pbp_df, atp_df, threshold)

            # Store performance metrics
            strategy_performance[strategy_name] = {
                "matches_found": len(results),
                "match_rate": len(results) / len(pbp_df) * 100 if len(pbp_df) > 0 else 0,
                "avg_score": results["score"].mean() if len(results) > 0 else 0,
                "processing_time": elapsed,
                "cached": False,
            }

            # Add strategy column and append to all results
            if len(results) > 0:
                results["strategy"] = strategy_name
                all_results.append(results)

            print(f"✅ {strategy_name} completed successfully")

        except Exception as e:
            print(f"❌ {strategy_name} failed: {e}")
            strategy_performance[strategy_name] = {
                "matches_found": 0,
                "match_rate": 0,
                "avg_score": 0,
                "processing_time": 0,
                "cached": False,
                "error": str(e),
            }

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
    else:
        combined_results = pd.DataFrame()

    # Print summary
    _print_experiment_summary(strategy_performance)

    return combined_results


def _create_matcher(strategy_name: str, config: dict, date_window_days: int = 30, use_tournament_normalization: bool = False):
    """Create appropriate matcher instance based on strategy name."""
    try:
        # Add date_window_days and tournament normalization to the config
        config_with_options = {**config, "date_window_days": date_window_days, "use_tournament_normalization": use_tournament_normalization}

        if strategy_name == "fuzzy":
            return FuzzyMatcher(**config_with_options)
        elif strategy_name == "embedding":
            return EmbeddingMatcher(**config_with_options)
        else:
            print(f"⚠️ Unknown strategy: {strategy_name}")
            return None
    except Exception as e:
        print(f"⚠️ Failed to create {strategy_name} matcher: {e}")
        return None


def _print_experiment_summary(strategy_performance: dict):
    """Print experiment summary with cache status."""
    print(f"\n{'='*80}")
    print("MATCHING EXPERIMENT SUMMARY")
    print(f"{'='*80}")

    # Sort strategies by match rate
    sorted_strategies = sorted(strategy_performance.items(), key=lambda x: x[1].get("match_rate", 0), reverse=True)

    print(f"\n{'Strategy':<15} {'Matches':<10} {'Rate':<8} {'Avg Score':<10} {'Time':<8} {'Source':<8}")
    print("-" * 70)

    for strategy, metrics in sorted_strategies:
        matches = metrics.get("matches_found", 0)
        rate = metrics.get("match_rate", 0)
        avg_score = metrics.get("avg_score", 0)
        time_taken = metrics.get("processing_time", 0)
        cached = metrics.get("cached", False)
        source = "Cached" if cached else "Fresh"

        print(f"{strategy:<15} {matches:<10,} {rate:<8.1f}% {avg_score:<10.1f} {time_taken:<8.1f}s {source:<8}")

        if "error" in metrics:
            print(f"  ❌ Error: {metrics['error']}")

    # Best strategy
    if sorted_strategies:
        best_strategy, best_metrics = sorted_strategies[0]
        print(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy.upper()}")
        print(f"   Match Rate: {best_metrics.get('match_rate', 0):.1f}%")
        print(f"   Matches Found: {best_metrics.get('matches_found', 0):,}")
        print(f"   Average Score: {best_metrics.get('avg_score', 0):.1f}")
        if best_metrics.get("cached", False):
            print("   🚀 Results loaded from cache (instant)")

    # Cache statistics
    cached_count = sum(1 for _, metrics in sorted_strategies if metrics.get("cached", False))
    if cached_count > 0:
        print(f"\n💾 Cache Performance: {cached_count}/{len(sorted_strategies)} strategies loaded from cache")
        total_fresh_time = sum(metrics.get("processing_time", 0) for _, metrics in sorted_strategies if not metrics.get("cached", False))
        if total_fresh_time > 0:
            print(f"   Time saved: {total_fresh_time:.1f}s (strategies that would have run fresh)")


if __name__ == "__main__":
    # Example usage
    print("🎾 Tennis Data Matching Module")
    print("This module provides flexible matching strategies for PBP integration.")
    print("Import and use the matchers in your data pipeline.")
    print("\nFor caching utilities, see: data_pipeline.caching")
    print("  - get_cache_info()")
    print("  - precompute_embeddings_for_datasets()")
    print("  - clear_embedding_cache()")
    print("  - clear_matching_results_cache()")
