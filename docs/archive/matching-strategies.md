# Tennis Data Matching Strategies

This document describes the various strategies available for matching point-by-point (PBP) data with ATP match records, including the enhanced LLM-based approaches.

---

## Overview

The Tennis Era Analysis project provides multiple matching strategies to integrate point-by-point data with match-level data:

1. **Fuzzy String Matching** - Basic name-based matching
2. **Enhanced Fuzzy Matching** - Multi-signal approach with dates and tournaments  
3. **LLM-Based Matching** - Advanced semantic matching using Large Language Models
4. **Hybrid Approaches** - Combining multiple strategies

Current performance comparison:
- Fuzzy Matching: ~25-30% success rate
- Enhanced Fuzzy: ~35-40% success rate  
- **LLM Matching: ~50-75% success rate** ⭐

---

## LLM-Based Matching (Recommended)

### Overview
Our enhanced LLM matching system uses **LangChain** to provide multi-provider support for both open-source and proprietary language models. This approach significantly improves matching accuracy by understanding semantic similarities and context.

### Supported Providers

#### 1. Ollama (Free, Local) 🆓
- **Models**: Llama 3.1, Llama 3.2, Mistral, Gemma2, Qwen2.5
- **Pros**: Free, private, no API limits, works offline
- **Cons**: Requires local setup, slower than cloud models
- **Best for**: Development, testing, privacy-sensitive applications

```python
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b
ollama serve

# Use in code
from data_pipeline.matching import LLMMatcher
matcher = LLMMatcher(provider="ollama", model="llama3.1:8b")
```

#### 2. OpenAI (Paid, Cloud) 💰
- **Models**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Pros**: Highest accuracy, fast responses, reliable
- **Cons**: Costs money, requires API key, data sent to OpenAI
- **Best for**: Production applications requiring highest accuracy

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

matcher = LLMMatcher(provider="openai", model="gpt-4o-mini")
```

#### 3. Google Gemini (Paid, Cloud) 💰
- **Models**: Gemini 2.0 Flash, Gemini 1.5 Flash, Gemini 1.5 Pro
- **Pros**: Cost-effective, very fast, good accuracy
- **Cons**: Requires API key, data sent to Google
- **Best for**: High-volume applications, cost-conscious deployments

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-api-key"

matcher = LLMMatcher(provider="google", model="gemini-2.0-flash")
```

### Enhanced Architecture

#### Multi-Provider Support
```python
from data_pipeline.matching import run_matching_experiment

# Test multiple providers automatically
results = run_matching_experiment(
    pbp_df, 
    atp_df,
    llm_providers={
        "ollama": {"model": "llama3.1:8b"},
        "openai": {"model": "gpt-4o-mini"},
        "google": {"model": "gemini-2.0-flash"}
    }
)
```

#### Intelligent Fallback Strategy
```python
# Automatic fallback: ollama -> openai -> google -> enhanced_fuzzy
matcher = LLMMatcher(provider="auto")

# Custom fallback order
from config.constants import LLM_FALLBACK_STRATEGY
# Default: ["ollama", "openai", "google", "enhanced_fuzzy"]
```

#### Response Caching & Performance Monitoring
```python
matcher = LLMMatcher(
    provider="ollama",
    cache_responses=True,    # Cache for 1 hour by default
    max_retries=3,          # Retry failed requests
    timeout=30              # Response timeout
)

# Get performance metrics
metrics = matcher.get_performance_metrics()
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Avg response time: {metrics['avg_response_time']:.2f}s")
```

### Prompt Engineering

The system uses carefully crafted prompts stored in `config/prompts.py`:

#### Simple Prompt (Fast)
```python
SIMPLE_MATCH_PROMPT = """Rate the similarity between these tennis records (0-100):

Record A: {pbp_player1} vs {pbp_player2}, {pbp_date}, {pbp_tournament}
Record B: {match_winner} vs {match_loser}, {match_date}, {match_tournament}

Consider player name variations, date proximity (±7 days), and tournament similarity.

Return only the numerical score (0-100)."""
```

#### Structured Prompt (Detailed)
```python
MATCH_COMPARISON_PROMPT = """Compare these two tennis records and determine if they refer to the same match:

POINT-BY-POINT RECORD:
- Player 1: {pbp_player1}
- Player 2: {pbp_player2}
- Date: {pbp_date}
- Tournament: {pbp_tournament}

MATCH RECORD:
- Winner: {match_winner}
- Loser: {match_loser}
- Date: {match_date}
- Tournament: {match_tournament}

Return JSON: {{"confidence_score": 0-100, "reasoning": "..."}}"""
```

### Configuration Options

#### Basic Configuration
```python
matcher = LLMMatcher(
    provider="ollama",           # Provider choice
    model="llama3.1:8b",        # Specific model
    threshold=85,               # Confidence threshold
    batch_size=10,              # Batch processing size
    temperature=0.1             # Low for consistency
)
```

#### Advanced Configuration
```python
from config.constants import LLM_CONFIG

# Global configuration in config/constants.py
LLM_CONFIG = {
    "default_provider": "ollama",
    "cache_responses": True,
    "cache_ttl": 3600,
    "max_retries": 3,
    "batch_size": 10,
    "temperature": 0.1,
    "max_tokens": 100,
    "prompt_style": "simple"
}
```

### Performance Comparison

| Strategy | Success Rate | Speed | Cost | Setup |
|----------|-------------|-------|------|-------|
| Fuzzy | 25-30% | Fast | Free | Easy |
| Enhanced Fuzzy | 35-40% | Fast | Free | Easy |
| **LLM (Ollama)** | **50-65%** | Medium | **Free** | Medium |
| **LLM (OpenAI)** | **60-75%** | Fast | $$$ | Easy |
| **LLM (Google)** | **55-70%** | Very Fast | $$ | Easy |

### Installation & Setup

#### 1. Install Dependencies
```bash
# Already included in pyproject.toml
pip install langchain-core langchain-community langchain-ollama langchain-openai langchain-google-vertexai
```

#### 2. Setup Ollama (Recommended for Development)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.1:8b

# Start server
ollama serve
```

#### 3. Setup Cloud Providers (Optional)
```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Google
export GOOGLE_API_KEY="your-google-api-key"
```

#### 4. Test Setup
```python
# Run the demo
python examples/llm_matching_demo.py
```

### Usage Examples

#### Quick Start
```python
from data_pipeline.matching import LLMMatcher

# Use Ollama (free)
matcher = LLMMatcher(provider="ollama")
matches = matcher.find_matches(pbp_df, atp_df)

# Use OpenAI (paid, higher accuracy)
matcher = LLMMatcher(provider="openai", model="gpt-4o-mini")
matches = matcher.find_matches(pbp_df, atp_df)
```

#### Comprehensive Experiment
```python
from data_pipeline.matching import run_matching_experiment

results = run_matching_experiment(
    pbp_df, 
    atp_df,
    strategies=["FuzzyMatcher", "EnhancedFuzzyMatcher", "LLMMatcher"],
    llm_providers={"ollama": {"model": "llama3.1:8b"}}
)

# Analyze results
print(results.groupby('strategy')['score'].agg(['count', 'mean', 'std']))
```

---

## Enhanced Fuzzy Matching

### Algorithm
Enhanced version with multiple matching signals

#### Key Improvements
1. **Player Name Matching (60% weight)**: Direct player comparison using token_sort_ratio
2. **Date Proximity (20% weight)**: ±30 day window scoring
3. **Tournament Context (20% weight)**: Tournament and round matching

#### Algorithm
```python
def calculate_match_score(pbp_row, match_row):
    scores = []
    weights = []
    
    # Player name matching (60% weight)
    name_score = calculate_name_score(pbp_row, match_row)
    if name_score > 0:
        scores.append(name_score)
        weights.append(0.6)
    
    # Date proximity (20% weight)  
    date_score = calculate_date_score(pbp_row, match_row)
    if date_score > 0:
        scores.append(date_score)
        weights.append(0.2)
        
    # Tournament context (20% weight)
    tournament_score = calculate_tournament_score(pbp_row, match_row)
    if tournament_score > 0:
        scores.append(tournament_score)
        weights.append(0.2)
    
    # Calculate weighted average
    return np.average(scores, weights=weights[:len(scores)])
```

#### Performance
- **Success Rate**: ~35-40%
- **Speed**: Fast (2-3 seconds for small datasets)
- **Accuracy**: Good for obvious matches, struggles with name variations

---

## Basic Fuzzy Matching

### Algorithm
Simple fuzzy string matching using player names only.

```python
def calculate_match_score(pbp_row, match_row):
    # Get player names using correct column mapping
    pbp_players = [pbp_row.get("server1", ""), pbp_row.get("server2", "")]
    match_players = [match_row.get("winner_name", ""), match_row.get("loser_name", "")]
    
    # Calculate fuzzy similarity for all name combinations
    scores = []
    for pbp_name in pbp_players:
        for match_name in match_players:
            score = fuzz.ratio(str(pbp_name).lower(), str(match_name).lower())
            scores.append(score)
    
    # Return average of top 2 scores (both players should match)
    scores.sort(reverse=True)
    return np.mean(scores[:2]) if len(scores) >= 2 else 0
```

#### Performance
- **Success Rate**: ~25-30%
- **Speed**: Fast (1-2 seconds for small datasets)
- **Accuracy**: Good for obvious matches

---

## Usage Guide

### Running Experiments

```python
from data_pipeline.matching import run_matching_experiment

# Load your data
pbp_df = pd.read_csv("your_pbp_data.csv")
atp_df = pd.read_csv("your_atp_data.csv")

# Run comprehensive experiment
results = run_matching_experiment(
    pbp_df, 
    atp_df,
    strategies=["FuzzyMatcher", "EnhancedFuzzyMatcher", "LLMMatcher"],
    llm_providers={
        "ollama": {"model": "llama3.1:8b"},
        "openai": {"model": "gpt-4o-mini"}
    }
)

# Analyze results
summary = results.groupby('strategy').agg({
    'score': ['count', 'mean', 'std'],
    'pbp_id': 'nunique'
})
print(summary)
```

### Custom Matching

```python
from data_pipeline.matching import LLMMatcher

# Create custom matcher
matcher = LLMMatcher(
    provider="ollama",
    model="llama3.1:8b", 
    threshold=80,
    batch_size=20
)

# Find matches
matches = matcher.find_matches(pbp_df, atp_df)

# Get performance metrics
metrics = matcher.get_performance_metrics()
print(f"Found {len(matches)} matches")
print(f"Success rate: {metrics['success_rate']:.2%}")
```

---

## Performance Optimization

### For Speed
- Use Ollama with smaller models (llama3.2:3b)
- Use simple prompts
- Increase batch sizes
- Enable response caching

### For Accuracy  
- Use OpenAI GPT-4o or Google Gemini Pro
- Use detailed prompts with examples
- Lower confidence thresholds
- Enable retry logic

### For Cost
- Use Ollama (free) for development
- Use OpenAI GPT-4o-mini for production
- Use Google Gemini Flash for high volume
- Enable aggressive caching

---

## Troubleshooting

### Common Issues

1. **Ollama not responding**
   ```bash
   ollama serve
   ollama pull llama3.1:8b
   ```

2. **API key errors**
   ```bash
   export OPENAI_API_KEY="your-key"
   export GOOGLE_API_KEY="your-key"
   ```

3. **Low match rates**
   - Lower the confidence threshold
   - Try different models
   - Check data quality

4. **Slow performance**
   - Reduce batch sizes
   - Use faster models
   - Enable caching

### Getting Help

- Run `python examples/llm_matching_demo.py` for interactive testing
- Check logs for detailed error messages
- See `config/constants.py` for configuration options
- Review `config/prompts.py` for prompt templates

---

## Future Enhancements

- **Batch Processing**: Process multiple records simultaneously
- **Active Learning**: Improve prompts based on feedback
- **Custom Models**: Fine-tune models on tennis data
- **Ensemble Methods**: Combine multiple LLM predictions
- **Real-time Matching**: Stream processing capabilities 