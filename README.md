# Tennis Analysis

Tennis Analysis is a comprehensive data pipeline and analytical framework for analyzing tennis player performance across different eras of the game. The project integrates point-by-point data with match-level statistics to provide deep insights into how tennis has evolved over time.

## Core Features

- **Data Cleaning**: Cleaning process removing duplicates, invalid data, and outliers
- **Data Standardization**: Clean and standardize ATP match and point-by-point datasets
- **Tournament Normalization**: Standardize tournament names for improved matching accuracy
- **Player-Match Transformation**: Convert match-level data to player-centric views
- **High-Performance Matching**: Advanced fuzzy and embedding-based matching with persistent caching
- **Era Analysis**: Statistical analysis of tennis evolution across different time periods
- **Career Age Curves**: Mixed-effects modeling of player peak ages and career trajectories across eras
- **Yearly Evolution Trends**: Granular year-by-year analysis with change point detection and phase identification
- **Interactive UI**: Streamlit-based web interface for visual exploration and analysis

## Interactive Web Interface

Launch the interactive analysis dashboard:

```bash
# Start the web interface
poetry run streamlit run ui/home.py
```

**Available Features:**
- **Age Curves Analysis** - Interactive peak age analysis and career trajectories
- **Era Analysis** - Comprehensive cross-era performance comparison with dynamic charts
- **Yearly Trends** - Year-over-year performance evolution and trend analysis

## Quick Start

### Basic Usage
```bash
# Run complete pipeline with optimized caching
python main.py

# Run with tournament normalization for better data quality
python main.py --tournament-normalization

# Run specific phases
python main.py --phase 1  # Cleaning + Standardization
python main.py --phase 2  # Transformation  
python main.py --phase 3  # Matching (optimized)
python main.py --phase 4  # Analysis

# Performance optimization
python main.py --precompute  # Precompute embeddings for maximum speed
python main.py --demo       # Quick cache performance demo
```

### Cache Management
```bash
# Clear caches when needed
python main.py --clear-cache all
```

## Performance Features

The matching system includes advanced caching for dramatic performance improvements:

- **Embedding Cache**: Persistent player name embeddings (10-50x faster)
- **Result Cache**: Complete matching results cached (100-1000x faster on subsequent runs)
- **Preprocessing**: Batch operations and vectorized calculations

## Problem Statement

Tennis performance metrics from different eras are not directly comparable due to:
- Equipment evolution (racquet technology, string materials)
- Training methodology advances
- Court surface standardization changes
- Tactical evolution of the game

## Solution

This pipeline processes raw ATP data through integrated cleaning and standardization to produce 51,806 cleaned matches (2005-2024) and 11,859 cleaned point-by-point records with era-normalized performance metrics.

### Key Components
- **Era Classification**: Four distinct periods based on game evolution patterns
- **Z-Score Normalization**: Removes temporal bias for fair cross-era comparisons
- **Player-Match Transformation**: Converts match-centric to player-centric data structure
- **Embedding Data Integration**: Links point-by-point records using vector similarity matching
- **Historical Context**: ATP ranking integration for match importance assessment

## Tennis Eras

- **Classic** (2005-2010): Federer dominance period
- **Transition** (2011-2015): Big 4 competitive balance
- **Modern** (2016-2020): NextGen emergence
- **Current** (2021-2024): New generation establishment

## Dataset Output

**Enhanced Dataset**: 116,162 player-match rows with 77 columns
- 18 era-normalized features using z-score transformation
- 99.3% ATP ranking coverage for historical context
- 92.4% point-by-point integration via embedding matching (enhanced v2024.12)

## Installation

```bash
git clone <repository-url>
cd Tennis_Era_Analysis
poetry install
poetry shell
```

## Embedding Setup (Optional)

For enhanced name matching accuracy:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download embedding model (670MB)
ollama pull mxbai-embed-large:latest

# Start Ollama service
ollama serve
```

## Documentation

- **[Technical Implementation Guide](docs/technical-implementation.md)**: Complete pipeline architecture and data schemas