"""
Tennis Era Analysis - Yearly Trends Page
Interactive analysis of year-over-year tennis evolution and performance trends.
"""

import streamlit as st


def render_yearly_trends_page():
    """Render the complete yearly trends analysis page."""

    try:
        # Page header
        st.markdown("# 📊 Yearly Trends Analysis")
        st.markdown("### Year-over-Year Evolution of Professional Tennis")
        st.markdown("---")

        # Introduction
        st.markdown("## 🎯 About Yearly Trends Analysis")
        st.markdown("""
        Explore the granular evolution of professional tennis through year-by-year analysis. 
        This section examines performance metrics trends, identifies significant transition points, 
        and reveals the phases of tennis evolution over the past two decades.
        
        **Key Features:**
        
        - **Year-over-Year Trends** - Statistical analysis of performance metric evolution
        - **Change Point Detection** - Identification of significant shifts in playing styles
        - **Evolution Phases** - Distinct periods of tennis development
        - **Interactive Visualizations** - Zoom, hover, and explore temporal patterns
        """)

        # Import components with error handling
        try:
            from ui.components.yearly_trends import (
                display_evolution_phases,
                display_performance_trends,
                display_trend_summary,
                display_yearly_overview,
            )

            component_import_success = True
        except Exception as e:
            st.error(f"Error importing yearly trends components: {e}")
            component_import_success = False

        if not component_import_success:
            st.stop()

        # Navigation tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Performance Trends", "🎯 Trend Summary", "🏟️ Evolution Phases"])

        with tab1:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
            try:
                with st.container():
                    display_yearly_overview()
            except Exception as e:
                st.error(f"Error in Yearly Overview: {e}")
                import traceback

                st.code(traceback.format_exc())
            st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
            try:
                with st.container():
                    display_performance_trends()
            except Exception as e:
                st.error(f"Error in Performance Trends: {e}")
                import traceback

                st.code(traceback.format_exc())
            st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
            try:
                with st.container():
                    display_trend_summary()
            except Exception as e:
                st.error(f"Error in Trend Summary: {e}")
                import traceback

                st.code(traceback.format_exc())
            st.markdown("</div>", unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
            try:
                with st.container():
                    display_evolution_phases()
            except Exception as e:
                st.error(f"Error in Evolution Phases: {e}")
                import traceback

                st.code(traceback.format_exc())
            st.markdown("</div>", unsafe_allow_html=True)

        # Research insights section
        st.markdown("---")
        st.markdown("## 💡 Key Research Insights")

        insight_col1, insight_col2, insight_col3 = st.columns(3)

        with insight_col1:
            st.markdown(
                """
            <div class="tennis-card">
                <h5>📈 Trend Detection</h5>
                <p>Advanced statistical methods identify significant changes in playing styles, 
                revealing when tennis underwent major evolutionary shifts.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with insight_col2:
            st.markdown(
                """
            <div class="tennis-card">
                <h5>🔄 Change Points</h5>
                <p>Piecewise regression analysis pinpoints specific years when multiple 
                performance metrics shifted simultaneously, marking transition periods.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with insight_col3:
            st.markdown(
                """
            <div class="tennis-card">
                <h5>⏳ Evolution Phases</h5>
                <p>Tennis evolution occurs in distinct phases, each characterized by 
                unique performance patterns and strategic developments.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Development roadmap
        st.markdown("---")
        st.markdown("## 🚧 Advanced Features (Coming Soon)")

        roadmap_col1, roadmap_col2 = st.columns(2)

        with roadmap_col1:
            st.info(
                """
                **🔮 Predictive Modeling**
                - Forecast future tennis trends
                - Predict next evolution phase
                - Model performance metric trajectories
                """
            )

        with roadmap_col2:
            st.info(
                """
                **🎾 Advanced Analytics**
                - Surface-specific yearly trends
                - Player cohort analysis
                - Tournament-level evolution patterns
                """
            )

        # Footer with methodological notes
        st.markdown("---")
        st.markdown(
            """
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #228B22;">
            <h6>📋 Methodology Notes</h6>
            <p><strong>Data Aggregation:</strong> Yearly means with minimum sample size requirements<br>
            <strong>Trend Analysis:</strong> Linear regression with R² significance testing<br>
            <strong>Change Detection:</strong> Piecewise regression with improvement thresholds<br>
            <strong>Phase Identification:</strong> Clustering of change points by temporal proximity</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Critical error in Yearly Trends page: {e}")
        import traceback

        st.code(traceback.format_exc())


if __name__ == "__main__":
    render_yearly_trends_page()
