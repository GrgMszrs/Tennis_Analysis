"""
Tennis Era Analysis - Home Page
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
    st.sidebar.markdown("## 🎾 Navigation")

    page = st.sidebar.selectbox(
        "Select Analysis:",
        ["🏠 Home", "📈 Age Curves", "🏟️ Era Analysis", "📊 Yearly Trends"],
        format_func=lambda x: x.split(" ", 1)[1],  # Remove emoji for cleaner display
    )

    # Add some spacing and info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Quick Stats")
    st.sidebar.markdown("**Status:** ✅ Active")
    st.sidebar.markdown("**Data:** ATP Tour")

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

    return page.split(" ", 1)[1]  # Return without emoji


def display_era_badges(eras):
    """Display era badges with custom styling."""
    st.markdown("**Available Eras:**")
    era_colors = {"Classic": "era-classic", "Transition": "era-transition", "Modern": "era-modern", "Current": "era-current"}

    badge_html = ""
    for era in eras:
        if era != "Unknown":
            color_class = era_colors.get(era, "era-current")
            badge_html += f'<span class="era-badge {color_class}">{era}</span>'

    st.markdown(badge_html, unsafe_allow_html=True)


def display_surface_badges(surfaces):
    """Display surface badges with custom styling."""
    st.markdown("**Available Surfaces:**")
    surface_colors = {"Hard": "surface-hard", "Clay": "surface-clay", "Grass": "surface-grass", "Carpet": "surface-carpet"}

    badge_html = ""
    for surface in surfaces:
        color_class = surface_colors.get(surface, "surface-hard")
        badge_html += f'<span class="surface-badge {color_class}">{surface}</span>'

    st.markdown(badge_html, unsafe_allow_html=True)


def main():
    """Main home page function."""

    # Page configuration
    st.set_page_config(page_title="Tennis Analysis", page_icon="🎾", layout="wide", initial_sidebar_state="expanded")

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

    # Home page content
    st.markdown("# 🎾 Tennis Era Analysis")
    st.markdown("### Interactive Analysis of Professional Tennis Evolution")
    st.markdown("---")

    # Hero section
    st.markdown(
        """
    <div style="padding: 1.5rem; border: 2px solid #228B22; border-radius: 10px; background: linear-gradient(135deg, #f0f8f0, #e8f5e8); margin: 1rem 0;">
        <h4>🏆 Welcome to Tennis Era Analysis</h4>
        <p>Explore the evolution of professional tennis through comprehensive data analysis spanning over 20 years of ATP Tour history. 
        Discover how playing styles, performance metrics, and player characteristics have evolved across different tennis eras.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load and display data summary
    st.markdown("## 📊 Dataset Overview")

    with st.spinner("🔄 Loading dataset summary..."):
        summary = get_data_summary()

    if summary:
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📈 Total Records", f"{summary['total_rows']:,}", help="Total number of player-match records in the dataset")

        with col2:
            st.metric("👥 Unique Players", f"{summary['unique_players']:,}", help="Number of distinct players in the dataset")

        with col3:
            st.metric("🎾 Unique Matches", f"{summary['unique_matches']:,}", help="Number of distinct matches analyzed")

        with col4:
            years = summary["years"]
            st.metric("📅 Year Range", f"{years[0]}-{years[1]}", help="Time period covered by the dataset")

        # Additional details in cards
        st.markdown("## 🏟️ Dataset Details")

        col1, col2 = st.columns(2)

        with col1:
            display_era_badges(summary["eras"])

        with col2:
            display_surface_badges(summary["surfaces"])

        # Enhanced features info
        if summary.get("enhanced_features"):
            enhanced = summary["enhanced_features"]
            st.markdown("## 🚀 Enhanced Analytics")

            feat_col1, feat_col2, feat_col3 = st.columns(3)

            with feat_col1:
                st.markdown(
                    f"""
                <div style="padding: 1rem; border: 1px solid #ddd; border-radius: 8px; background: #f8f9fa;">
                    <h5>📊 Normalized Metrics</h5>
                    <p><strong>{enhanced["z_score_metrics"]}</strong> z-score normalized performance indicators for cross-era comparisons</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with feat_col2:
                st.markdown(
                    f"""
                <div style="padding: 1rem; border: 1px solid #ddd; border-radius: 8px; background: #f8f9fa;">
                    <h5>🏆 Ranking Features</h5>
                    <p><strong>{enhanced["ranking_metrics"]}</strong> historical ranking and career progression metrics</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with feat_col3:
                st.markdown(
                    """
                <div style="padding: 1rem; border: 1px solid #ddd; border-radius: 8px; background: #f8f9fa;">
                    <h5>⚡ Performance</h5>
                    <p>Optimized data processing with <strong>caching</strong> and <strong>indexing</strong></p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    else:
        st.error("❌ Unable to load dataset. Please check the data pipeline.")

    # Navigation guide
    st.markdown("## 🧭 Navigation Guide")

    nav_col1, nav_col2, nav_col3 = st.columns(3)

    with nav_col1:
        st.markdown(
            """
        <div style="padding: 1rem; border: 1px solid #ddd; border-radius: 8px; background: #f8f9fa; margin: 0.5rem 0;">
            <h5>📈 Age Curves</h5>
            <p><strong>✅ Ready</strong></p>
            <p>• Interactive career trajectories<br>
            • Peak age analysis by era<br>
            • Zoom, pan, hover features<br>
            • Native chart rendering</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with nav_col2:
        st.markdown(
            """
        <div style="padding: 1rem; border: 1px solid #ddd; border-radius: 8px; background: #f8f9fa; margin: 0.5rem 0;">
            <h5>🏟️ Era Analysis</h5>
            <p><strong>✅ Ready</strong></p>
            <p>• Interactive performance trends<br>
            • Dynamic surface heatmaps<br>
            • Era comparison charts<br>
            • Champion analysis</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with nav_col3:
        st.markdown(
            """
        <div style="padding: 1rem; border: 1px solid #ddd; border-radius: 8px; background: #f8f9fa; margin: 0.5rem 0;">
            <h5>📊 Yearly Trends</h5>
            <p><strong>✅ Ready</strong></p>
            <p>• Year-over-year evolution<br>
            • Trend change detection<br>
            • Game phase analysis<br>
            • Interactive temporal charts</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #7F8C8D; font-size: 0.9em;">
        <p>🎾 Tennis Era Analysis | Built with Streamlit & ❤️ for Tennis Analytics</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
