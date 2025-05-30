"""
Tennis Era Analysis - Caching System
High-performance caching for embeddings and matching results.
"""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from config.constants import EMBEDDING_CONFIG, EMBEDDING_MODEL


class EmbeddingCache:
    """Persistent cache for player name embeddings with batch processing."""

    def __init__(self, cache_dir: str = "data/cache/embeddings", model: str = EMBEDDING_MODEL):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.base_url = EMBEDDING_CONFIG["base_url"]
        self.api_url = f"{self.base_url}/api/embeddings"

        # Cache files
        self.embeddings_file = self.cache_dir / f"player_embeddings_{model.replace('/', '_')}.pkl"
        self.metadata_file = self.cache_dir / "embedding_metadata.json"

        # Load existing cache
        self.embeddings = self._load_cache()
        self.metadata = self._load_metadata()

    def _load_cache(self) -> Dict[str, np.ndarray]:
        """Load existing embeddings from disk."""
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, "rb") as f:
                    embeddings = pickle.load(f)
                print(f"📂 Loaded {len(embeddings):,} cached embeddings")
                return embeddings
            except Exception as e:
                print(f"⚠️ Failed to load embedding cache: {e}")
        return {}

    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load metadata: {e}")
        return {"model": self.model, "created_at": time.time(), "last_updated": time.time()}

    def _save_cache(self):
        """Save embeddings and metadata to disk."""
        try:
            # Save embeddings
            with open(self.embeddings_file, "wb") as f:
                pickle.dump(self.embeddings, f)

            # Update and save metadata
            self.metadata.update({"model": self.model, "last_updated": time.time(), "embedding_count": len(self.embeddings)})
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)

            print(f"💾 Saved {len(self.embeddings):,} embeddings to cache")
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")

    def get_embeddings(self, names: List[str], batch_size: int = 50) -> Dict[str, np.ndarray]:
        """Get embeddings for a list of names, computing missing ones in batches."""
        # Normalize names
        normalized_names = [str(name).strip().lower() for name in names if str(name) != "nan"]
        unique_names = list(set(normalized_names))

        # Find missing embeddings
        missing_names = [name for name in unique_names if name not in self.embeddings]

        if missing_names:
            print(f"🔄 Computing {len(missing_names):,} new embeddings...")
            self._compute_batch_embeddings(missing_names, batch_size)
            self._save_cache()

        # Return requested embeddings
        return {name: self.embeddings.get(name, np.zeros(384)) for name in normalized_names}

    def _compute_batch_embeddings(self, names: List[str], batch_size: int = 50):
        """Compute embeddings in batches for efficiency."""
        total_batches = (len(names) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(names), batch_size), desc="Computing embeddings", total=total_batches):
            batch_names = names[i : i + batch_size]

            # Process individual calls (batch API not implemented in most Ollama setups)
            for name in batch_names:
                try:
                    embedding = self._get_single_embedding(name)
                    self.embeddings[name] = embedding
                except Exception as e:
                    print(f"⚠️ Failed to get embedding for '{name}': {e}")
                    self.embeddings[name] = np.zeros(384)

    def _get_single_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text string."""
        response = requests.post(self.api_url, json={"model": self.model, "prompt": text}, timeout=5)
        response.raise_for_status()
        embedding_data = response.json()

        if "embedding" in embedding_data:
            return np.array(embedding_data["embedding"], dtype=np.float32)
        else:
            raise ValueError(f"No embedding in response for '{text}'")


class MatchingResultsCache:
    """Persistent cache for matching results with automatic cleanup."""

    def __init__(self, cache_dir: str = "data/cache/matching_results"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_dataset_fingerprint(self, pbp_df: pd.DataFrame, match_df: pd.DataFrame) -> str:
        """Generate a fingerprint for the datasets to detect changes."""
        pbp_hash = hashlib.md5(str(pbp_df.shape).encode() + str(pbp_df.columns.tolist()).encode()).hexdigest()[:8]
        match_hash = hashlib.md5(str(match_df.shape).encode() + str(match_df.columns.tolist()).encode()).hexdigest()[:8]
        return f"{pbp_hash}_{match_hash}"

    def _get_cache_filename(self, strategy: str, fingerprint: str, threshold: float) -> Path:
        """Generate cache filename for matching results."""
        return self.cache_dir / f"matches_{strategy}_{fingerprint}_t{threshold:.0f}.pkl"

    def _clean_old_cache_files(self, strategy: str, current_fingerprint: str, threshold: float):
        """Remove old cache files for the same strategy/threshold but different datasets."""
        current_file = self._get_cache_filename(strategy, current_fingerprint, threshold)
        pattern = f"matches_{strategy}_*_t{threshold:.0f}.pkl"

        # Find all files matching the pattern
        matching_files = list(self.cache_dir.glob(pattern))

        for file_path in matching_files:
            if file_path != current_file:
                try:
                    file_path.unlink()
                    print(f"🗑️  Cleaned old cache file: {file_path.name}")
                except Exception as e:
                    print(f"⚠️ Failed to clean old cache file {file_path.name}: {e}")

    def has_cached_results(self, strategy: str, pbp_df: pd.DataFrame, match_df: pd.DataFrame, threshold: float) -> bool:
        """Check if cached results exist for the given parameters."""
        fingerprint = self._get_dataset_fingerprint(pbp_df, match_df)
        cache_file = self._get_cache_filename(strategy, fingerprint, threshold)
        return cache_file.exists()

    def load_cached_results(self, strategy: str, pbp_df: pd.DataFrame, match_df: pd.DataFrame, threshold: float) -> Optional[pd.DataFrame]:
        """Load cached matching results if available."""
        fingerprint = self._get_dataset_fingerprint(pbp_df, match_df)
        cache_file = self._get_cache_filename(strategy, fingerprint, threshold)

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    results = pickle.load(f)
                print(f"📂 Loaded cached {strategy} results: {len(results):,} matches")
                return results
            except Exception as e:
                print(f"⚠️ Failed to load cached results: {e}")
        return None

    def save_results(self, results: pd.DataFrame, strategy: str, pbp_df: pd.DataFrame, match_df: pd.DataFrame, threshold: float):
        """Save matching results to cache and clean up old files."""
        fingerprint = self._get_dataset_fingerprint(pbp_df, match_df)

        # Clean up old cache files for this strategy/threshold combination
        self._clean_old_cache_files(strategy, fingerprint, threshold)

        # Save new results
        cache_file = self._get_cache_filename(strategy, fingerprint, threshold)

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)
            print(f"💾 Cached {len(results):,} {strategy} results: {cache_file.name}")
        except Exception as e:
            print(f"⚠️ Failed to save results to cache: {e}")


def clear_embedding_cache(model: str = EMBEDDING_MODEL):
    """Clear the embedding cache for a specific model."""
    cache = EmbeddingCache(model=model)
    try:
        if cache.embeddings_file.exists():
            cache.embeddings_file.unlink()
        if cache.metadata_file.exists():
            cache.metadata_file.unlink()
        print(f"✅ Cleared embedding cache for model: {model}")
    except Exception as e:
        print(f"⚠️ Failed to clear embedding cache: {e}")


def clear_matching_results_cache():
    """Clear all cached matching results."""
    cache_dir = Path("data/cache/matching_results")
    if cache_dir.exists():
        try:
            import shutil

            shutil.rmtree(cache_dir)
            print("✅ Cleared all matching results cache")
        except Exception as e:
            print(f"⚠️ Failed to clear matching results cache: {e}")
    else:
        print("ℹ️ No matching results cache found")


def precompute_embeddings_for_datasets(pbp_df: pd.DataFrame, match_df: pd.DataFrame, model: str = EMBEDDING_MODEL) -> int:
    """
    Precompute embeddings for all player names in the datasets.
    This is useful to run once to populate the cache before running multiple experiments.
    """
    print("🔄 Precomputing embeddings for all datasets...")

    embedding_cache = EmbeddingCache(model=model)

    # Extract all unique names
    pbp_names = []
    for col in ["server1", "server2"]:
        if col in pbp_df.columns:
            names = pbp_df[col].dropna().astype(str).tolist()
            pbp_names.extend([n for n in names if n != "nan"])

    match_names = []
    for col in ["winner_name", "loser_name"]:
        if col in match_df.columns:
            names = match_df[col].dropna().astype(str).tolist()
            match_names.extend([n for n in names if n != "nan"])

    all_names = list(set(pbp_names + match_names))
    print(f"   Found {len(all_names):,} unique player names")

    # Get embeddings (this will compute and cache missing ones)
    embedding_cache.get_embeddings(all_names)

    print("✅ Precomputation complete!")
    return len(all_names)
