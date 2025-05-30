# Phase 2: Reshape to "One Row Per Player-Match"

## 🎯 **Strategy: Winner/Loser → Player/Opponent Format**

**Why Reshape?** Player comparisons across eras become trivial when every row represents "Player A vs Player B on Date X" instead of "Winner vs Loser"

## 📊 **Transformation Plan**

### **Test Dataset: January 2020** (manageable ~500 matches)
```sql
SELECT * FROM atp_matches_standardized 
WHERE tourney_date >= '2020-01-01' AND tourney_date < '2020-02-01'
```

### **Core Transformation Logic:**

| Original Column | Becomes | Side | Keep? | Notes |
|-----------------|---------|------|-------|-------|
| `winner_id` | `player_id` | W | ✅ | Core identifier |
| `loser_id` | `opp_id` | W | ✅ | Opponent |
| `loser_id` | `player_id` | L | ✅ | Player (flipped row) |
| `winner_id` | `opp_id` | L | ✅ | Opponent (flipped row) |
| All `w_*` stats | `aces`, `df`, `first_in`, etc. | W | ✅ | Rename consistently |
| All `l_*` stats | `aces`, `df`, `first_in`, etc. | L | ✅ | Same stats, flipped |
| `winner_rank` | `player_rank` | W | ✅ | Gold standard metric |
| `loser_rank` | `opp_rank` | W | ✅ | Context |
| `winner_rank` | `player_rank` | L | ✅ | Flipped |
| `loser_rank` | `opp_rank` | L | ✅ | Flipped |

### **New Derived Columns:**
- `won_match` (BOOLEAN): TRUE for winner side, FALSE for loser side  
- `side` (CHAR): 'W' or 'L' for debugging/validation
- `rank_diff` (INT): `player_rank - opp_rank` (negative = underdog)

### **Demographic Table Strategy:** 
Move to separate `player_bio` table:
- `player_id`, `hand`, `height`, `country`, `birth_date`
- Join when needed, don't duplicate in main table

## 🔧 **Implementation Approach**

### **Step 1: Test on January 2020**
```python
# Filter test data  
jan_2020 = atp_matches_std[
    (atp_matches_std['tourney_date'] >= '2020-01-01') & 
    (atp_matches_std['tourney_date'] < '2020-02-01')
].copy()

print(f"Test dataset: {len(jan_2020)} matches → {len(jan_2020) * 2} player-match rows")
```

### **Step 2: Create Winner Rows**
```python
winner_rows = jan_2020.rename(columns={
    'winner_id': 'player_id',
    'loser_id': 'opp_id', 
    'winner_rank': 'player_rank',
    'loser_rank': 'opp_rank',
    # All w_ columns → stat columns
    'w_ace': 'aces',
    'w_df': 'double_faults', 
    # ... etc
}).assign(
    won_match=True,
    side='W'
)
```

### **Step 3: Create Loser Rows**  
```python
loser_rows = jan_2020.rename(columns={
    'loser_id': 'player_id',
    'winner_id': 'opp_id',
    'loser_rank': 'player_rank', 
    'winner_rank': 'opp_rank',
    # All l_ columns → stat columns
    'l_ace': 'aces',
    'l_df': 'double_faults',
    # ... etc  
}).assign(
    won_match=False,
    side='L'
)
```

### **Step 4: Combine & Validate**
```python
player_matches = pd.concat([winner_rows, loser_rows], ignore_index=True)

# Validation checks
assert len(player_matches) == len(jan_2020) * 2
assert player_matches['won_match'].sum() == len(jan_2020)  # Half won
assert player_matches.groupby('match_id').size().eq(2).all()  # 2 rows per match
```

## 📈 **Analysis Capabilities Unlocked**

### **Player-Year Aggregations:**
```python
player_year_stats = player_matches.groupby([
    'player_id', 
    player_matches['tourney_date'].dt.year
]).agg({
    'won_match': ['count', 'sum'],  # matches_played, matches_won
    'aces': 'sum',
    'double_faults': 'sum',
    # ... etc
})
```

### **Rolling 52-Week Windows:**
```sql
SELECT player_id, tourney_date,
       SUM(won_match) OVER (
           PARTITION BY player_id 
           ORDER BY tourney_date 
           RANGE BETWEEN INTERVAL 52 WEEK PRECEDING AND CURRENT ROW
       ) as wins_52w
FROM player_matches
```

### **Era Comparisons:**
```python
# Compare 2005-2010 vs 2015-2020
early_era = player_matches[player_matches['tourney_date'].dt.year.between(2005, 2010)]
modern_era = player_matches[player_matches['tourney_date'].dt.year.between(2015, 2020)]
```

## ⚠️ **Critical Success Factors**

1. **Test thoroughly on January 2020** before full dataset
2. **Validate row counts** at every step (2× matches = player-match rows)
3. **Handle the join challenge** between ATP matches and PBP data
4. **Create comprehensive validation suite** 
5. **Memory management** for full 116K+ row dataset

## 🔄 **PBP Integration Strategy**

Since ATP PBP has different IDs, create joining logic:
```python
# Fuzzy match on: date + player names + tournament
pbp_matched = match_pbp_to_atp_matches(
    atp_pbp_std, 
    player_matches,
    on=['date', 'player_names', 'tournament']
)
```

## 📁 **Output Files**
- `player_matches_jan2020.csv` (test)
- `player_matches_full.csv` (full dataset) 
- `player_bio.csv` (demographic lookup)
- `validation_report.txt`

---

**Next Step:** Implement test on January 2020 data to validate the approach! 