# Tennis Era Analysis Database Schema

## Overview
This document describes the structure and content of the Tennis Era Analysis dataset, which transforms match-level data into player-match rows for comprehensive tennis era analysis.

## Dataset Transformation
- **Input**: ATP match data (58,081 matches, 2005-2024)
- **Output**: Player-match dataset (116,162 rows, 2 per match)
- **Transformation**: Each tennis match creates two rows - one from each player's perspective

---

## Core Data Structure

### Match Identification
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `match_id` | String | Unique identifier for each tennis match | "2024-555-M-1" |
| `match_date` | Date | Date when match was played | "2024-01-15" |
| `match_date_int` | Integer | Date as integer for sorting | 20240115 |
| `match_number` | Integer | Sequential match number within tournament | 64 |

### Tournament Information
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `tournament_id` | Integer | Unique tournament identifier | 555 |
| `tournament_name` | String | Official tournament name | "Australian Open" |
| `tournament_level` | String | Tournament category | "Grand Slam" |
| `tournament_round` | String | Round of tournament | "4th Round" |
| `court_surface` | String | Playing surface | "Hard" |
| `draw_size` | Integer | Number of players in draw | 128 |

### Match Details
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `final_score` | String | Complete match score | "6-4 6-3 6-2" |
| `sets_format` | String | Match format | "Best of 5" |
| `match_duration_minutes` | Integer | Total match duration | 142 |

---

## Player Data (Dual Perspective)

### Player Identity
| Column | Type | Description | Notes |
|--------|------|-------------|--------|
| `player_id` | Integer | ATP player ID | Primary player in this row |
| `player_name` | String | Player's full name | "Rafael Nadal" |
| `opponent_id` | Integer | Opponent's ATP player ID | |
| `opponent_name` | String | Opponent's full name | |

### Match Outcome
| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `won_match` | Boolean | Did this player win? | True/False |
| `side` | String | Original data perspective | "Winner"/"Loser" |

### Rankings & Seeding
| Column | Type | Description | Range |
|--------|------|-------------|--------|
| `player_ranking` | Integer | ATP ranking | 1-2000+ |
| `player_ranking_points` | Integer | ATP ranking points | 0-12000+ |
| `player_seed` | Integer | Tournament seeding | 1-32 (if seeded) |
| `opponent_ranking` | Integer | Opponent's ATP ranking | |
| `opponent_ranking_points` | Integer | Opponent's ranking points | |
| `opponent_seed` | Integer | Opponent's tournament seed | |

### Player Attributes
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `player_age` | Float | Player age at match | 25.3 |
| `player_handedness` | String | Playing hand | "Right"/"Left" |
| `player_height_cm` | Integer | Height in centimeters | 185 |
| `player_country_code` | String | IOC country code | "ESP" |
| `player_entry_type` | String | Entry method | "Direct"/"Qualifier"/"Wildcard" |

---

## Match Statistics

### Serve Statistics (Player)
| Column | Type | Description | Usage |
|--------|------|-------------|--------|
| `aces` | Integer | Aces served | Power analysis |
| `double_faults` | Integer | Double faults | Consistency analysis |
| `serve_points` | Integer | Total service points | Volume measure |
| `first_serves_in` | Integer | First serves in | Accuracy measure |
| `first_serve_points_won` | Integer | Points won on 1st serve | Effectiveness |
| `second_serve_points_won` | Integer | Points won on 2nd serve | |
| `service_games` | Integer | Service games played | |
| `break_points_faced` | Integer | Break points faced | Pressure situations |
| `break_points_saved` | Integer | Break points saved | Clutch performance |

### Opponent Serve Statistics
All serve statistics are mirrored with `opponent_` prefix to provide complete match context.

---

## Era-Focused Derived Metrics

### Serve Dominance Analysis
| Metric | Formula | Purpose | Era Relevance |
|--------|---------|---------|---------------|
| `serve_dominance_index` | `aces / service_games` | Measure serve power evolution | Power era vs finesse |
| `first_serve_effectiveness` | `first_serve_points_won / first_serves_in` | Precision improvement | Modern accuracy trends |
| `service_hold_rate` | `1 - (opponent_breaks / service_games)` | Game control evolution | Defensive improvements |

### Mental Toughness Metrics
| Metric | Formula | Purpose | Era Relevance |
|--------|---------|---------|---------------|
| `break_point_save_rate` | `break_points_saved / break_points_faced` | Clutch performance | Mental game evolution |

### Return Game Evolution
| Metric | Formula | Purpose | Era Relevance |
|--------|---------|---------|---------------|
| `return_effectiveness` | `opponent_return_points_won / opponent_serve_points` | Return strategy evolution | Aggressive return trends |

### Competitive Context
| Metric | Formula | Purpose | Era Relevance |
|--------|---------|---------|---------------|
| `ranking_advantage` | `opponent_ranking - player_ranking` | Upset analysis | Ranking parity trends |
| `age_advantage` | `opponent_age - player_age` | Experience vs youth | Career longevity trends |

### Match Dynamics
| Metric | Formula | Purpose | Era Relevance |
|--------|---------|---------|---------------|
| `match_intensity` | `total_points / match_duration_minutes` | Pace evolution | Speed-up rule impact |

### Era Classification
| Column | Type | Description | Periods |
|--------|------|-------------|---------|
| `era` | Category | Tennis era classification | Classic (2005-2010)<br>Transition (2011-2015)<br>Modern (2016-2020)<br>Current (2021+) |

---

## Data Integration

### Point-by-Point Integration
| Column | Type | Description | Coverage |
|--------|------|-------------|----------|
| `pbp_match_id` | String | Linked PBP match identifier | ~27.6% of matches |
| `match_score` | Float | Matching confidence score | 75-100 |
| `strategy` | String | Matching algorithm used | "enhanced_fuzzy"/"llm" |

### Integration Methods
1. **Enhanced Fuzzy Matching**: Tournament name, date, players, duration similarity
2. **LLM Matching** (optional): AI-powered semantic matching for complex cases
3. **Composite Matching**: Weighted combination of multiple strategies

---

## Data Quality Metrics

| Aspect | Coverage | Notes |
|--------|----------|--------|
| **Serve Statistics** | 85%+ | High coverage for post-2010 data |
| **Rankings** | 95%+ | Nearly complete ranking data |
| **Player Demographics** | 90%+ | Height/handedness occasionally missing |
| **PBP Integration** | 27.6% | Limited to 2011-2017 period |

---

## Usage Examples

### Era Comparison Query
```sql
SELECT era, 
       AVG(serve_dominance_index) as avg_serve_power,
       AVG(first_serve_effectiveness) as avg_precision
FROM player_matches 
WHERE serve_dominance_index IS NOT NULL
GROUP BY era
ORDER BY era
```

### Player Evolution Analysis
```sql
SELECT player_name,
       YEAR(match_date) as year,
       AVG(break_point_save_rate) as mental_toughness
FROM player_matches 
WHERE player_name = 'Rafael Nadal'
  AND break_point_save_rate IS NOT NULL
GROUP BY player_name, YEAR(match_date)
ORDER BY year
```

### Surface-Era Interaction
```sql
SELECT era, court_surface,
       AVG(match_intensity) as avg_pace,
       COUNT(*) as matches
FROM player_matches
GROUP BY era, court_surface
ORDER BY era, court_surface
```

---

## Technical Notes

### Performance Considerations
- **Row Count**: 116,162 rows (manageable for most analysis tools)
- **Memory Usage**: ~150MB in CSV format
- **Indexing**: Recommend indexing on `player_id`, `match_date`, `era`

### Missing Data Handling
- **Serve Stats**: Use `IS NOT NULL` filters for serve-based analysis
- **Rankings**: Missing rankings typically indicate unranked players
- **Derived Metrics**: Handle division by zero in custom calculations

### Data Validation Rules
- Each `match_id` appears exactly twice (winner + loser perspective)
- `won_match` is True for exactly half of all rows
- `serve_dominance_index` should be positive and reasonable (<5.0)
- Era assignments are mutually exclusive and comprehensive 