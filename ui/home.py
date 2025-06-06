"""
Tennis Analysis - Home Page
Main landing page for the Streamlit UI application.
"""

from pathlib import Path

import streamlit as st

from ui.components.data_loader import get_data_summary


def load_custom_css():
    """Load custom CSS styling."""
    css_file = Path(__file__).parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def create_sidebar():
    """Create sidebar navigation."""
    st.sidebar.markdown("## Navigation")

    page = st.sidebar.selectbox(
        "Select Analysis:",
        ["Home", "Age Curves", "Era Analysis", "Yearly Trends", "Date Analysis"],
    )

    # Add some spacing and info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    st.sidebar.markdown("**Status:** Active")
    st.sidebar.markdown("**Source:** ATP Tour")

    # Get actual data coverage dynamically
    try:
        summary = get_data_summary()
        if summary and "years" in summary:
            year_start, year_end = summary["years"]
            st.sidebar.markdown(f"**Coverage:** {year_start}-{year_end}")
        else:
            st.sidebar.markdown("**Coverage:** Loading...")
    except Exception:
        st.sidebar.markdown("**Coverage:** 2005-2024")

    return page


def display_era_badges(eras):
    """Display era badges with custom styling."""
    st.markdown("**Available Eras:**")

    badge_html = ""
    for era in eras:
        if era != "Unknown":
            era_class = f"era-{era.lower()}"
            badge_html += f'<span class="era-badge {era_class}">{era}</span>'

    st.markdown(badge_html, unsafe_allow_html=True)


def display_surface_badges(surfaces):
    """Display surface badges with custom styling."""
    st.markdown("**Available Surfaces:**")

    badge_html = ""
    for surface in surfaces:
        surface_class = f"surface-{surface.lower()}"
        badge_html += f'<span class="surface-badge {surface_class}">{surface}</span>'

    st.markdown(badge_html, unsafe_allow_html=True)


def main():
    """Main home page function."""

    # Page configuration
    st.set_page_config(page_title="Tennis Analysis", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

    # Load custom styling
    load_custom_css()

    # Navigation - sidebar only
    selected_page = create_sidebar()

    # Route to different pages
    if selected_page == "Age Curves":
        from ui.modules.age_curves import render_age_curves_page

        render_age_curves_page()
        return
    elif selected_page == "Era Analysis":
        from ui.modules.era_analysis import render_era_analysis_page

        render_era_analysis_page()
        return
    elif selected_page == "Yearly Trends":
        from ui.modules.yearly_trends import render_yearly_trends_page

        render_yearly_trends_page()
        return
    elif selected_page == "Date Analysis":
        from ui.modules.date_analysis import render_date_analysis_page

        render_date_analysis_page()
        return

    # Home page content
    st.markdown("# Tennis Era Analysis")
    st.markdown("### Statistical Analysis of Professional Tennis Evolution")
    st.markdown("---")

    # Hero section
    st.markdown(
        """
    <div class="highlight-section">
        <h4>Tennis Performance Analytics Platform</h4>
        <p>Comprehensive statistical analysis of ATP Tour data spanning 20+ years. Analyze performance metrics evolution, 
        career trajectories, and playing style changes across tennis eras using standardized statistical methods and 
        interactive visualizations.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load and display data summary
    st.markdown("## Dataset Overview")

    with st.spinner("Loading dataset summary..."):
        summary = get_data_summary()

    if summary:
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{summary['total_rows']:,}", help="Player-match records in dataset")

        with col2:
            st.metric("Unique Players", f"{summary['unique_players']:,}", help="Distinct players analyzed")

        with col3:
            st.metric("Unique Matches", f"{summary['unique_matches']:,}", help="Distinct matches analyzed")

        with col4:
            years = summary["years"]
            st.metric("Time Coverage", f"{years[0]}-{years[1]}", help="Temporal range of dataset")

        # Additional details in cards
        st.markdown("## Data Classification")

        col1, col2 = st.columns(2)

        with col1:
            display_era_badges(summary["eras"])

        with col2:
            display_surface_badges(summary["surfaces"])

        # Enhanced features info
        if summary.get("enhanced_features"):
            enhanced = summary["enhanced_features"]
            st.markdown("## Feature Engineering")

            feat_col1, feat_col2, feat_col3 = st.columns(3)

            with feat_col1:
                st.markdown(
                    f"""
                <div class="insight-card">
                    <h5>Normalized Metrics</h5>
                    <p><strong>{enhanced["z_score_metrics"]}</strong> z-score normalized performance indicators enable cross-era statistical comparisons</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with feat_col2:
                st.markdown(
                    f"""
                <div class="insight-card">
                    <h5>Historical Rankings</h5>
                    <p><strong>{enhanced["ranking_metrics"]}</strong> ranking progression and career trajectory features</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with feat_col3:
                st.markdown(
                    """
                <div class="insight-card">
                    <h5>Performance Optimization</h5>
                    <p>Cached data processing with indexing for sub-second query response times</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    else:
        st.error("Unable to load dataset. Check data pipeline configuration.")

    # Navigation guide
    st.markdown("## Analysis Modules")

    nav_col1, nav_col2 = st.columns(2)
    nav_col3, nav_col4 = st.columns(2)

    with nav_col1:
        st.markdown(
            """
        <div class="analysis-card">
            <h5>Age Curves Analysis</h5>
            <p><strong>Status:</strong> Active</p>
            <ul>
                <li>Career trajectory modeling</li>
                <li>Peak age statistical analysis</li>
                <li>Interactive Plotly visualizations</li>
                <li>Era-based performance comparison</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with nav_col2:
        st.markdown(
            """
        <div class="analysis-card">
            <h5>Era Analysis</h5>
            <p><strong>Status:</strong> Active</p>
            <ul>
                <li>Performance metric trend analysis</li>
                <li>Surface-specific performance heatmaps</li>
                <li>Statistical era comparison</li>
                <li>Champion identification algorithms</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with nav_col3:
        st.markdown(
            """
        <div class="analysis-card">
            <h5>Yearly Trends</h5>
            <p><strong>Status:</strong> Active</p>
            <ul>
                <li>Time-series trend analysis</li>
                <li>Change point detection</li>
                <li>Evolution phase identification</li>
                <li>Multi-metric temporal modeling</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with nav_col4:
        st.markdown(
            """
        <div class="analysis-card">
            <h5>Date Analysis</h5>
            <p><strong>Status:</strong> Active</p>
            <ul>
                <li>Temporal pattern analysis</li>
                <li>Seasonal distribution modeling</li>
                <li>Match scheduling intensity</li>
                <li>Surface and era timeline trends</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Technical details
    st.markdown("---")
    st.markdown("## Technical Specifications")

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.markdown(
            """
        **Data Processing Pipeline:**
        - Raw ATP Tour match data ingestion
        - Feature engineering and normalization
        - Era classification based on temporal and stylistic patterns
        - Statistical validation and quality control
        """
        )

    with tech_col2:
        st.markdown(
            """
        **Analysis Framework:**
        - Plotly for interactive visualizations
        - Pandas for data manipulation and analysis
        - Streamlit for web interface
        - Cached computations for performance optimization
        """
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div class="footer">
        <p>Tennis Analysis | Statistical Computing Platform for Tennis Analytics</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
