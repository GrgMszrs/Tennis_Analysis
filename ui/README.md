# Tennis Era Analysis - Streamlit UI

Interactive web interface for tennis data analysis and visualization.

### File Structure

```
ui/
├── __init__.py             # Module initialization
├── home.py                 # Main landing page with navigation
├── README.md               # This documentation
├── assets/
│   └── style.css           # Custom tennis-themed CSS styling
├── components/
│   ├── __init__.py
│   ├── data_loader.py      # Cached data loading functions
│   ├── age_analysis.py     # Age curves analysis components
│   ├── era_analysis.py     # Era analysis components
│   └── chart_utils.py      # Plotly chart utilities
└── modules/
    ├── __init__.py
    ├── age_curves.py       # Age curves analysis page
    └── era_analysis.py     # Era analysis page
```

### Running the UI

```bash
# From project root
poetry run streamlit run ui/home.py
```

The UI will start at `http://localhost:8501

### Current Features ✅

1. **🏠 Home Page** - Enhanced landing page with:
   - Dataset overview and live statistics
   - Era and surface badges with custom styling
   - Quick navigation to analysis pages
   - Performance metrics display

2. **📈 Age Curves Analysis** - Complete implementation:
   - Interactive peak age analysis by era
   - Plotly box plots with hover details and zoom functionality
   - Era statistics comparison table
   - Player filtering and exploration
   - Career trajectory insights

3. **🏟️ Era Analysis** - Comprehensive analysis:
   - Era overview with statistics comparison
   - Interactive trend analysis with line plots
   - Dynamic surface performance heatmaps
   - Era champions identification
   - Tabbed interface for organized viewing

### Technology Stack

- **Frontend**: Streamlit with custom CSS
- **Charts**: Plotly for native interactive rendering
- **Data**: Cached pandas DataFrames for performance
- **Styling**: Custom tennis-themed color schemes
- **Backend**: Integration with existing analysis modules

### Navigation Guide

The sidebar provides navigation between:
- **🏠 Home** - Dataset overview and quick stats
- **📈 Age Curves** - Player career analysis and peak age evolution
- **🏟️ Era Analysis** - Comprehensive era comparison and trends
- **📊 Yearly Trends** - *Coming soon - temporal evolution analysis*

### Next Development Steps

1. **📊 Yearly Trends Page** - Integrate the yearly evolution analysis
2. **👤 Individual Player Profiles** - Deep-dive player analysis pages
3. **🔍 Advanced Filtering** - Enhanced search and filter capabilities
4. **📁 Export Features** - Download charts and data as PDF/CSV
5. **📱 Mobile Enhancements** - Improved mobile experience

### Dependencies

- `streamlit` - Web application framework
- `plotly` - Interactive visualization library
- `pandas` - Data manipulation
- Existing project analysis modules for backend data processing