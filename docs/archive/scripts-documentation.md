# Analysis Scripts

This directory contains standalone analysis scripts for the Tennis Era Analysis project.

## 📊 Available Scripts

### `data_quality_analysis.py`
Comprehensive data quality assessment tool.

**Purpose**: Analyze data standardization quality, null patterns, categorical mappings, and date conversions.

**Usage**:
```bash
poetry run python scripts/data_quality_analysis.py
```

**Features**:
- Null value analysis (before/after standardization)
- Categorical standardization validation
- Date conversion success rates
- Data consistency checks
- Coverage analysis
- Professional quality report

### `date_visualization.py`
Tennis match date pattern visualization tool.

**Purpose**: Create comprehensive visualizations of match date distributions and temporal patterns.

**Usage**:
```bash
poetry run python scripts/date_visualization.py
```

**Features**:
- Annual match count trends
- Monthly distribution timeline
- Seasonal pattern analysis
- Surface distribution over time
- Data coverage assessment
- High-quality plot exports

### `tournament_normalization_demo.py`
Tournament name standardization demonstration tool.

**Purpose**: Demonstrate tournament name normalization capabilities and analyze tournament overlap between datasets.

**Usage**:
```bash
poetry run python scripts/tournament_normalization_demo.py
```

**Features**:
- Tournament name normalization examples
- PBP-ATP tournament overlap analysis
- Standardization effectiveness metrics
- Tournament filtering demonstrations
- Mapping table generation

### `joinability_heatmap_analysis.py`
PBP-ATP data joinability analysis and visualization tool.

**Purpose**: Analyze temporal and player overlap patterns between point-by-point and ATP match datasets.

**Usage**:
```bash
poetry run python scripts/joinability_heatmap_analysis.py
```

**Features**:
- Player presence analysis by grouping strategies
- Temporal overlap heatmap visualization
- Tournament mismatch pattern analysis
- Joinability rate calculations
- Comprehensive visual reports

### `optimize_fuzzy_matching.py`
Fuzzy date matching optimization tool.

**Purpose**: Test different date windows to find optimal PBP-ATP matching parameters.

**Usage**:
```bash
poetry run python scripts/optimize_fuzzy_matching.py
```

**Features**:
- Date window optimization (1-30 days)
- Match rate analysis and visualization
- Diminishing returns analysis
- Optimal parameter recommendations
- Performance vs accuracy trade-off analysis

### `investigate_data_overlap.py`
Data overlap investigation and analysis tool.

**Purpose**: Deep dive analysis of player and temporal overlap between datasets.

**Usage**:
```bash
poetry run python scripts/investigate_data_overlap.py
```

**Features**:
- Player name overlap analysis
- Temporal coverage comparison
- Data quality assessment
- Missing data pattern identification
- Coverage gap analysis

## 🎯 When to Use These Scripts

### Data Quality Analysis
Run this script when you need to:
- Validate data standardization results
- Investigate potential data quality issues
- Generate reports for stakeholders
- Verify pipeline integrity

### Date Visualization
Run this script when you need to:
- Understand temporal patterns in tennis data
- Identify seasonal trends
- Visualize data coverage gaps
- Create presentation-ready charts

### Tournament Normalization Demo
Run this script when you need to:
- Understand tournament name standardization
- Analyze tournament overlap between datasets
- Evaluate normalization effectiveness
- Generate tournament mapping tables

### Joinability Heatmap Analysis
Run this script when you need to:
- Analyze PBP-ATP data overlap patterns
- Understand temporal coverage differences
- Identify matching bottlenecks
- Generate joinability visualizations

### Fuzzy Matching Optimization
Run this script when you need to:
- Optimize date window parameters
- Analyze matching performance trade-offs
- Find optimal matching configurations
- Understand diminishing returns in matching

### Data Overlap Investigation
Run this script when you need to:
- Deep dive into player overlap patterns
- Investigate missing data causes
- Analyze temporal coverage gaps
- Understand dataset limitations

## 📁 Output Locations

These scripts save their outputs to:
- **Visualizations**: `data/output/plots/` (date patterns, joinability heatmaps, optimization charts)
- **Analysis Reports**: Console output with detailed formatting and statistics
- **Tournament Mappings**: Generated mapping tables and normalization examples
- **Performance Metrics**: Optimization results and recommendations

## 🔧 Requirements

These scripts require:
- Standardized data files in `data/cleaned_refactored/`
- Project dependencies installed via Poetry
- Python path configuration (handled automatically)
- Sufficient disk space for plot generation
- Optional: Internet connection for some analysis features

## 📋 Integration with Main Pipeline

These scripts are **supplementary analysis tools** and are not part of the core data pipeline. They provide:

- **Data Quality Insights**: Validate and understand your data
- **Optimization Guidance**: Find optimal parameters for matching
- **Visualization Tools**: Create publication-ready charts and reports
- **Debugging Support**: Investigate data issues and pipeline performance

They can be run independently at any time after data standardization.

For the main pipeline, use:
```bash
poetry run python main.py
``` 