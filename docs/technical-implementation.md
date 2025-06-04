# Tennis Era Analysis - Technical Implementation Guide

## Overview

Comprehensive technical documentation for the Tennis Era Analysis pipeline, covering data processing architecture, schema specifications, and implementation details for processing 58,081 ATP matches and 11,859 point-by-point records to generate era-normalized tennis performance metrics.

### Core Components
- **Player-Match Transformation**: Dual-perspective data structure (winner/loser views)
- **Era Normalization**: Z-score standardization across temporal periods
- **Embedding Matching**: Semantic integration of point-by-point records using vector embeddings
- **Ranking Integration**: Historical ATP ranking context (2000-2024)

---

## Data Pipeline Architecture & File Outputs

### Phase 1: Standardization
**Input**: Raw ATP match data, point-by-point records
**Process**: Date normalization, numeric type conversion, categorical standardization
**Output Files**:
- `atp_matches_standardized.csv` (15MB) - 58,081 matches with standardized schema
- `atp_pbp_standardized.csv` (3.7MB) - 11,859 PBP records with consistent formatting

**Critical Fix (v2024.12)**: Resolved circular dependency in data loading that caused 100% date conversion failures. Pipeline now properly loads cleaned data for standardization and standardized data for matching.

**Schema Standardization**:
- ATP Matches: 51 columns including match metadata, player demographics, match statistics
- PBP Data: 15 columns with player names, match details, point-by-point sequences
- Common identifiers: `match_id`, standardized date columns, player names
- **Date Quality**: 100% successful conversion (58,081/58,081 ATP matches, 11,859/11,859 PBP records)

### Phase 2: Enhanced Transformation
**Input**: Standardized match data
**Process**: Era classification, z-score calculation, ranking integration, player-match conversion
**Output Files**:
- `atp_player_match_enhanced.csv` (84MB) - **PRIMARY DATASET** with full enhancements  
- `atp_player_match_enhanced_summary.json` - Metadata and feature summary

**Result**: 58,081 matches → 116,162 player-match rows with 77 columns

### Phase 3: Embedding Matching (Enhanced v2024.12)
**Input**: Standardized ATP and PBP datasets
**Process**: Vector embedding-based semantic matching with optimized date window fuzzy matching
**Output**: Integrated records with cosine similarity scores ≥85%
**Performance**: 92.4% match rate (10,955/11,859 PBP records successfully matched)
**Processing Time**: ~90 seconds for full dataset
**Cache Files**: `data/cache/embeddings/` and `data/cache/matching_results/`

**Major Improvements**:
- **Date Window Optimization**: 7-day fuzzy matching window (79.1% → 92.4% with embeddings)
- **Data Pipeline Fix**: Eliminated circular dependency causing 0% match rates
- **Performance Enhancement**: 50x improvement in processing speed with embedding caching

### Phase 4: Analysis
**Input**: Enhanced player-match dataset (`atp_player_match_enhanced.csv`)
**Process**: Cross-era statistical analysis and visualization generation
**Output**: Research-ready analysis reports in `data/output/`

---

## Complete Data Schema

### Primary Dataset: `atp_player_match_enhanced.csv`
**Dimensions**: 116,162 rows × 77 columns

#### Column Groups & Schema:

**1. Match Metadata (13 columns)**:
- `match_id` - Unique match identifier
- `tourney_date` - Tournament date (YYYY-MM-DD)
- `tourney_name` - Tournament name
- `surface` - Court surface (Hard, Clay, Grass, Carpet)
- `round` - Tournament round (R128, R64, R32, R16, QF, SF, F)
- `best_of` - Match format (3 or 5 sets)
- `score` - Final match score
- `minutes` - Match duration in minutes
- `tourney_level` - Tournament category (G, M, A, D)
- `tourney_id` - Tournament identifier
- `match_num` - Match number within tournament
- `tourney_date_int` - Date as integer (YYYYMMDD)
- `era` - Era classification (Classic, Transition, Modern, Current)

**2. Temporal Context (1 column)**:
- `year` - Match year

**3. Player Identity (8 columns)**:
- `player_id` - ATP player ID
- `seed` - Tournament seeding
- `entry` - Entry type (direct, qualifier, wildcard)
- `player_name` - Player name
- `hand` - Handedness (R/L)
- `height` - Height in cm
- `country` - Country code
- `age` - Player age at match

**4. Current Rankings (2 columns)**:
- `rank` - ATP ranking at match time
- `rank_points` - ATP ranking points

**5. Match Statistics - Player (12 columns)**:
- `aces` - Aces served
- `double_faults` - Double faults
- `serve_points` - Total service points
- `first_serves_in` - First serves made
- `first_serves_won` - First serve points won
- `second_serves_won` - Second serve points won
- `service_games` - Service games played
- `break_points_saved` - Break points saved
- `break_points_faced` - Break points faced
- `ace_rate` - Aces per service game
- `df_rate` - Double faults per service game
- `first_serve_pct` - First serve percentage

**6. Calculated Percentages - Player (6 columns)**:
- `l_first_serve_pct` - Legacy first serve percentage
- `first_serve_win_pct` - First serve win percentage
- `l_first_serve_win_pct` - Legacy first serve win percentage
- `second_serve_win_pct` - Second serve win percentage
- `l_second_serve_win_pct` - Legacy second serve win percentage
- `break_point_save_pct` - Break point save percentage

**7. Return & Service Metrics (6 columns)**:
- `l_break_point_save_pct` - Legacy break point save percentage
- `return_win_pct` - Return points won percentage
- `l_return_win_pct` - Legacy return win percentage
- `service_dominance` - Service dominance metric
- `l_service_dominance` - Legacy service dominance
- `won_match` - Boolean: did player win this match

**8. Match Perspective (1 column)**:
- `player_type` - Perspective type (winner/loser)

**9. Winner Statistics (6 columns)**:
- `w_first_serve_pct` - Winner's first serve percentage
- `w_first_serve_win_pct` - Winner's first serve win percentage
- `w_second_serve_win_pct` - Winner's second serve win percentage
- `w_break_point_save_pct` - Winner's break point save percentage
- `w_return_win_pct` - Winner's return win percentage
- `w_service_dominance` - Winner's service dominance

**10. Opponent Information (4 columns)**:
- `opponent_name` - Opponent name
- `opponent_id` - Opponent ATP ID
- `opponent_rank` - Opponent ranking
- `opponent_rank_points` - Opponent ranking points

**11. Z-Score Normalization - Year Only (8 columns)**:
- `ace_rate_z_year` - Year-normalized ace rate
- `df_rate_z_year` - Year-normalized double fault rate
- `first_serve_pct_z_year` - Year-normalized first serve percentage
- `first_serve_win_pct_z_year` - Year-normalized first serve win percentage
- `second_serve_win_pct_z_year` - Year-normalized second serve win percentage
- `break_point_save_pct_z_year` - Year-normalized break point save percentage
- `return_win_pct_z_year` - Year-normalized return win percentage
- `service_dominance_z_year` - Year-normalized service dominance

**12. Z-Score Normalization - Year + Surface (8 columns)**:
- `ace_rate_z_year_surface` - Year+surface normalized ace rate
- `df_rate_z_year_surface` - Year+surface normalized double fault rate
- `first_serve_pct_z_year_surface` - Year+surface normalized first serve percentage
- `first_serve_win_pct_z_year_surface` - Year+surface normalized first serve win percentage
- `second_serve_win_pct_z_year_surface` - Year+surface normalized second serve win percentage
- `break_point_save_pct_z_year_surface` - Year+surface normalized break point save percentage
- `return_win_pct_z_year_surface` - Year+surface normalized return win percentage
- `service_dominance_z_year_surface` - Year+surface normalized service dominance

**13. Historical Rankings (2 columns)**:
- `historical_rank` - Historical ATP ranking (365-day window lookup)
- `historical_points` - Historical ATP ranking points

---

## Point-by-Point Data Schema

### Dataset: `atp_pbp_standardized.csv`
**Dimensions**: 11,859 rows × 15 columns

#### Schema:
- `pbp_id` - PBP-specific match identifier
- `date` - Original date string from source
- `tny_name` - Tournament abbreviation
- `tour` - Tour level (ATP)
- `draw` - Draw type (MAIN)
- `server1` - First player name (order-invariant)
- `server2` - Second player name (order-invariant)
- `winner` - Winner identifier (1 or 2)
- `pbp` - Full point-by-point sequence string
- `score` - Match final score
- `adf_flag` - Additional data flag
- `wh_minutes` - Match duration in minutes
- `parsed_date` - Parsed date format
- `date_standardized` - Standardized date (YYYY-MM-DD)
- `match_id` - Standardized match identifier (pbp_<pbp_id>)

**Data Quality**: 100% successful date standardization (11,859/11,859 records)

---

## Data Quality & Coverage Metrics

### Enhanced Dataset Quality:
- **Total Matches**: 58,081 → 116,162 player-match rows
- **Player Coverage**: 2,478 unique players (updated with latest data)
- **Temporal Coverage**: 2005-2024 (20 years)
- **Historical Ranking Coverage**: 99.3% (115,335/116,162 matches)
- **Z-score Validation**: Mean ≈ 0, std ≈ 1 for all normalized metrics

### PBP Integration Results (Enhanced v2024.12):
- **Total PBP Records**: 11,859
- **Temporal Overlap**: 2011-2017 (6 years vs ATP's 20 years)
- **Successful Matches**: 10,955 records (92.4%) - **Major improvement from 0%**
- **Average Similarity Score**: 99.6% - High confidence matches
- **Processing Speed**: 89.9 seconds for full dataset

### Date Window Optimization Analysis:
- **1-day window**: 30.5% match rate (3,620 matches)
- **3-day window**: 59.7% match rate (7,074 matches)
- **7-day window**: 79.1% match rate (9,377 matches) - **Optimal balance**
- **14-day window**: 80.8% match rate (9,580 matches) - Diminishing returns
- **30-day window**: 80.8% match rate (9,580 matches) - No further improvement

**Optimization Impact**: 7-day window provides 19.4 percentage point improvement over 3-day window with minimal risk of false positives.

### Pipeline Reliability Improvements:
- **Date Conversion Success**: 100% (previously 0% due to circular dependency)
- **Data Schema Consistency**: Standardized column names across pipeline
- **Error Handling**: Robust fallback mechanisms for API failures
- **Performance Monitoring**: Real-time progress tracking and ETA estimation

### Quality Assurance Metrics:
- **Data Integrity**: No duplicate match IDs, consistent player name formats
- **Temporal Validity**: All dates within expected ranges (2005-2024 ATP, 2011-2017 PBP)
- **Statistical Validation**: Z-score distributions meet normalization criteria
- **Coverage Validation**: >99% ranking coverage maintained after updates

### Known Data Limitations:
1. **PBP Date Range**: Limited to 2011-2017 vs ATP's 2005-2024
2. **Tournament Scope**: PBP focuses on major tournaments, ATP includes all levels
3. **Missing Records**: 7.6% PBP records remain unmatched due to temporal gaps or missing players
4. **Column Schema Differences**: Final vs regular standardized files use different naming conventions

---

## Era Classification System

**Classic Era (2005-2010)**: Federer dominance, traditional playing styles
**Transition Era (2011-2015)**: Big 4 competitive balance, tactical evolution  
**Modern Era (2016-2020)**: NextGen emergence, technology integration
**Current Era (2021-2024)**: New generation establishment

Classification based on match date with era-specific performance baselines for normalization.

---

## Z-Score Normalization Methodology

**Mathematical Foundation**: `z = (x - μ) / σ`

**Two Approaches**:
1. **Year-Only**: `z_year = (value - year_mean) / year_std`
   - Use case: Cross-surface comparisons within era
2. **Year+Surface**: `z_year_surface = (value - year_surface_mean) / year_surface_std`
   - Use case: Surface-specific specialist identification

**Normalized Metrics** (8 core × 2 approaches = 16 features):
- Ace rate, double fault rate, first serve percentage
- First/second serve win percentages
- Break point save percentage, return win percentage
- Service dominance (composite metric)

**Validation**: Mean ≈ 0, standard deviation ≈ 1 for all z-score features

---

## Player-Match Transformation

**Rationale**: Standard match data provides winner/loser statistics but lacks player-centric perspective for individual analysis.

**Implementation**: Each match generates two rows:
- Winner perspective: `won_match=True`, player stats as primary, opponent stats as secondary
- Loser perspective: `won_match=False`, player stats as primary, opponent stats as secondary

**Result**: 58,081 matches → 116,162 player-match records enabling consistent player-focused analysis.

---

## Enhanced Embedding Matching System (v2024.12)

**Problem**: Point-by-point records use inconsistent player names, tournament names, and date formats compared to ATP match data.

**Solution**: Ollama embedding-based semantic matching with optimized fuzzy date window.

**Technical Implementation**: 
- **Primary**: Ollama embedding model (`mxbai-embed-large:latest`) for vector-based name matching
- **Fallback**: Fuzzy string matching for reliability when embedding API unavailable
- **Date Optimization**: 7-day fuzzy window for optimal temporal matching

**Architecture Improvements**:
- **Separated Data Loading**: `load_raw_datasets()` for standardization, `load_standardized_datasets()` for matching
- **Proper Date Handling**: Fixed circular dependency causing 100% date conversion failures
- **Column Schema Alignment**: Ensured consistent naming between standardization and matching phases
- **Performance Caching**: Persistent embedding cache for 2,461+ unique player names

**Technical Approach**:
- Generate 1024-dimensional embeddings for player names using Ollama `/api/embeddings` endpoint
- Calculate cosine similarity between name vectors (0-1 scale, converted to 0-100 percentage)
- Cache embeddings for unique names to avoid recomputation
- Apply 85% similarity threshold for match acceptance
- Use ±7 day fuzzy date window for temporal alignment

**Performance Results (v2024.12)**:
- **Match Rate**: 92.4% (10,955/11,859 records) - **Massive improvement from 0%**
- **Average Confidence**: 99.6% similarity score
- **Processing Speed**: 89.9 seconds for full dataset
- **Temporal Precision**: 7-day window optimally balances accuracy vs coverage
- **Cache Efficiency**: 2,461 cached embeddings enable fast repeated matching

**Model Selection**: `mxbai-embed-large:latest` provides optimal accuracy/speed trade-off at 670MB disk usage.

---

## Historical Ranking Integration

**Data Source**: Jeff Sackmann ATP ranking repository (2,261,808 records, 2000-2024)

**Algorithm**: For each player-match, find closest ranking within 365-day window using minimum date difference.

**Coverage Results**:
- Player matching: 99.9% (2,119/2,121 players)
- Ranking coverage: 99.3% (115,335/116,162 matches)

**Purpose**: Enables upset analysis, match importance assessment, and ranking-context performance evaluation.

---

## Implementation Specifications

### Environment Requirements
- Python 3.11+
- Core: pandas, numpy, scipy, scikit-learn
- Embedding: requests (for Ollama API), numpy (for vector operations)
- Analysis: matplotlib, seaborn
- Matching: fuzzywuzzy, python-levenshtein

### Configuration Parameters (Optimized)
- **Date window**: 7 days (optimized from analysis)
- Minimum matches for z-score calculation: 10
- Ranking search window: 365 days
- Embedding similarity threshold: 85%
- Embedding batch size: 50
- Embedding cache size: 10,000 unique names

### Quality Assurance
- Dataset structure validation (116,162 rows, 77 columns)
- Z-score distribution validation (mean ≈ 0, std ≈ 1)
- Ranking coverage validation (>99%)
- Era distribution validation (4 eras present)
- **Date conversion validation**: 100% success rate (fixed in v2024.12)

### Performance Optimizations
- Chunk processing for memory management
- Embedding caching for repeated name lookups
- Multiprocessing for independent operations
- Pre-filtering before expensive operations
- Vector operations using NumPy for efficient similarity calculations
- **Optimized date window**: 7-day fuzzy matching for best performance/accuracy trade-off

---

## Research Applications

### Enabled Analyses
1. **Cross-Era Performance Comparison**: Era-adjusted metrics remove temporal bias
2. **Surface Specialization**: Surface-specific z-scores identify specialists
3. **Game Evolution Tracking**: Temporal trends in normalized performance metrics
4. **Upset Pattern Analysis**: Ranking-context upset identification and trends
5. **Peak Performance Assessment**: Fair comparison of player peaks across eras
6. **Point-by-Point Integration**: 92.4% of PBP records now linked to ATP match context

### Statistical Methodologies
- Hierarchical modeling for nested effects (player/tournament/era)
- Time series analysis for career trajectory modeling
- Survival analysis for career longevity studies
- Causal inference for performance factor identification

### Extension Opportunities
- Shot-level analysis integration (Hawk-Eye data)
- Injury context incorporation
- Weather condition effects
- Betting market efficiency analysis
- Social media sentiment correlation

---

## File Management & Caching Strategy

### Cache Organization:
- `data/cache/embeddings/` - Player name embeddings (persistent)
- `data/cache/matching_results/` - PBP-ATP matching results (persistent)

### Cleanup Commands:
- `python main.py --clear-cache embeddings` - Clear embedding cache
- `python main.py --clear-cache results` - Clear matching results cache  
- `python main.py --clear-cache all` - Clear all caches

### Pipeline Optimization:
- Use `--precompute` flag for maximum matching performance
- Use `--no-cache` flag for fresh results (slower)
- Era-specific datasets available for focused analysis
- Use `--force` flag to regenerate standardized data after schema changes