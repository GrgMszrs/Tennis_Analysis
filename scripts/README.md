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

## 📁 Output Locations

Both scripts save their outputs to:
- **Plots**: `data/output/`
- **Logs**: Console output with professional formatting

## 🔧 Requirements

These scripts require:
- Standardized data files in `data/cleaned_refactored/`
- Project dependencies installed via Poetry
- Python path configuration (handled automatically)

## 📋 Integration with Main Pipeline

These scripts are **supplementary analysis tools** and are not part of the core data pipeline. They can be run independently at any time after data standardization.

For the main pipeline, use:
```bash
poetry run python main.py
``` 