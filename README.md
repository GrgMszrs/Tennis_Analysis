# Tennis Era Analysis

A comprehensive data pipeline for analyzing tennis performance across different eras (2005-2024) using clean, modular architecture.

## 🎾 What This Does

This project processes **58,081 ATP matches** and **11,859 point-by-point records** to create analysis-ready datasets with era-focused derived metrics for cross-era tennis performance comparison.

## ✅ Current Status

**PRODUCTION READY** - Clean, modular pipeline with comprehensive analysis capabilities:

- ✅ **Phase 1**: Data standardization (58K+ matches processed)
- ✅ **Phase 2**: Player-match transformation with era metrics  
- ✅ **Phase 3**: Flexible PBP matching strategies
- ✅ **Phase 4**: Comprehensive era analysis and visualization
- ✅ **Clean Architecture**: Modular, testable, maintainable code

## 🏗️ Project Structure

```
Tennis_Era_Analysis/
├── config/                    # Configuration and constants
│   ├── __init__.py
│   └── constants.py          # All project constants and settings
├── data_pipeline/            # Data processing modules
│   ├── __init__.py
│   ├── standardization.py    # Phase 1: Data standardization
│   ├── transformation.py     # Phase 2: Player-match transformation
│   └── matching.py          # Phase 3: PBP matching strategies
├── analysis/                 # Analysis and visualization
│   ├── __init__.py
│   └── era_analysis.py      # Era comparison and statistics
├── scripts/                  # Standalone analysis tools
│   ├── data_quality_analysis.py  # Data quality assessment
│   ├── date_visualization.py     # Date pattern visualization
│   └── README.md            # Scripts documentation
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── helpers.py          # Common helper functions
├── tests/                   # Test files (to be added)
├── data/                    # Data files (gitignored)
│   ├── cleaned_refactored/  # Processed datasets
│   └── output/             # Analysis outputs
├── main.py                 # Main entry point
├── pyproject.toml         # Poetry configuration
└── .cursorrules          # Development guidelines
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Poetry (for dependency management)

### Installation
        ```bash
# Clone the repository
git clone <repository-url>
cd Tennis_Era_Analysis

# Install dependencies with Poetry
        poetry install

# Activate virtual environment
        poetry shell
        ```

### Running the Pipeline

        ```bash
# Run the complete pipeline
poetry run python main.py

# Run in test mode (faster)
poetry run python main.py --test

# Run specific phases
poetry run python main.py --phase 1  # Standardization only
poetry run python main.py --phase 2  # Transformation only
poetry run python main.py --phase 4  # Analysis only

# Skip PBP matching (faster)
poetry run python main.py --skip-matching

# Get help
poetry run python main.py --help
```

## 📊 Data Pipeline

### Phase 1: Standardization
- **Input**: Raw ATP match and PBP data
- **Process**: Date conversion, numeric standardization, categorical mapping
- **Output**: Standardized datasets with universal match IDs
- **Module**: `data_pipeline.standardization`

### Phase 2: Transformation  
- **Input**: Standardized match data
- **Process**: Era classification, derived metrics, player-match reshape
- **Output**: Analysis-ready player-match format with era metrics
- **Module**: `data_pipeline.transformation`

### Phase 3: Matching (Optional)
- **Input**: Standardized ATP and PBP data
- **Process**: Fuzzy matching, enhanced matching, optional LLM matching
- **Output**: Matched PBP records with confidence scores
- **Module**: `data_pipeline.matching`

### Phase 4: Analysis
- **Input**: Player-match data
- **Process**: Era statistics, trend analysis, champion identification
- **Output**: Comprehensive analysis report with visualizations
- **Module**: `analysis.era_analysis`

## 🏆 Key Features

### Era Definitions
- **Classic Era** (2005-2010): Early modern tennis
- **Transition Era** (2011-2015): Changing game dynamics  
- **Modern Era** (2016-2020): Current playing style emergence
- **Current Era** (2021-2024): Latest developments

### Derived Metrics
- **Service Analytics**: Ace rates, first serve effectiveness, service dominance
- **Pressure Situations**: Break point save percentages
- **Return Game**: Return win percentages and effectiveness
- **Era Comparisons**: Cross-era statistical analysis

### Matching Strategies
- **Fuzzy Matching**: Basic string similarity
- **Enhanced Fuzzy**: Multi-signal matching (names, dates, tournaments)
- **LLM Matching**: AI-powered matching (optional, requires OpenAI API)

## 📈 Analysis Capabilities

### Statistical Analysis
- Era-wise performance statistics
- Trend analysis across time periods
- Surface-specific performance comparison
- Player ranking and champion identification

### Visualizations
- Era comparison box plots
- Trend line analysis
- Performance distribution charts
- Surface-specific breakdowns

## 🛠️ Development

### Adding New Features
1. Follow the modular structure
2. Add constants to `config/constants.py`
3. Use type hints and docstrings
4. Update documentation
5. Write tests

### Code Style
- Follow PEP 8 guidelines
- Use Poetry for dependency management
- Implement clean code principles
- See `.cursorrules` for detailed guidelines

### Testing
        ```bash
# Run tests (when implemented)
poetry run pytest

# Test individual modules
poetry run python -m data_pipeline.standardization
poetry run python -m analysis.era_analysis
```

## 📁 Data Files

### Input Data (Required)
- `data/cleaned_refactored/atp_matches_cleaned.csv` - ATP match data
- `data/cleaned_refactored/atp_pbp_cleaned.csv` - Point-by-point data

### Output Data (Generated)
- `data/cleaned_refactored/atp_matches_standardized.csv` - Standardized matches
- `data/cleaned_refactored/player_match_data.csv` - Analysis-ready dataset
- `data/cleaned_refactored/player_match_data_*.csv` - Era-specific subsets

## 🔧 Configuration

All configuration is centralized in `config/constants.py`:
- Era definitions and date ranges
- File paths and data sources
- Processing parameters and thresholds
- Column mappings and transformations

## 📚 Documentation

- **README.md**: Project overview and usage
- **Module docstrings**: Detailed function documentation
- **.cursorrules**: Development guidelines and best practices
- **Type hints**: Function signatures and return types

## 🤝 Contributing

1. Follow the established project structure
2. Use Poetry for dependency management
3. Adhere to clean code principles
4. Update documentation with changes
5. Write tests for new functionality

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- ATP for providing match data
- Jeff Sackmann for tennis datasets
- Tennis analytics community for insights

---

**Ready to analyze tennis eras!** 🎾📊 

## 🔧 Analysis Scripts

For supplementary analysis and data quality checks, use the standalone scripts:

```bash
# Data quality assessment
poetry run python scripts/data_quality_analysis.py

# Date pattern visualization
poetry run python scripts/date_visualization.py
```

These scripts provide:
- **Data Quality Analysis**: Null patterns, categorical validation, date conversion checks
- **Date Visualization**: Temporal patterns, seasonal trends, coverage analysis
- **Professional Reports**: Console output and high-quality plot exports

See `scripts/README.md` for detailed documentation. 