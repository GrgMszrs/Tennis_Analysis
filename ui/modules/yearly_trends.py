"""
Tennis Analysis - Yearly Trends Page
Interactive analysis of year-over-year tennis evolution and performance trends.
"""

import streamlit as st


def render_yearly_trends_page():
    """Render the complete yearly trends analysis page."""

    try:
        # Page header
        st.markdown("# Yearly Trends Analysis")
        st.markdown("### Time-Series Analysis of Professional Tennis Evolution")
        st.markdown("---")

        # Introduction
        st.markdown("## Overview")
        st.markdown("""
        Statistical analysis of professional tennis evolution through year-by-year performance metrics. 
        Time-series analysis methods identify significant transitions, change points, and evolutionary 
        phases in tennis development over the past two decades.
        
        **Analysis Components:**
        
        - **Temporal Trends** - Linear regression analysis of performance metric evolution
        - **Change Point Detection** - Statistical identification of significant transition periods
        - **Evolution Phases** - Clustering analysis of distinct developmental periods
        - **Multi-Metric Modeling** - Comparative analysis of multiple performance indicators
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
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Performance Trends", "Trend Analysis", "Evolution Phases"])

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
        st.markdown("## Statistical Insights")

        insight_col1, insight_col2, insight_col3 = st.columns(3)

        with insight_col1:
            st.markdown(
                """
            <div class="insight-card">
                <h5>Trend Detection</h5>
                <p>Linear regression with R² significance testing identifies statistically 
                significant changes in performance metrics, with p-value thresholds ensuring 
                robust trend identification.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with insight_col2:
            st.markdown(
                """
            <div class="insight-card">
                <h5>Change Point Analysis</h5>
                <p>Piecewise regression analysis identifies years with significant metric 
                transitions, using improvement thresholds and temporal clustering to 
                validate change points.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with insight_col3:
            st.markdown(
                """
            <div class="insight-card">
                <h5>Phase Identification</h5>
                <p>Clustering algorithms group change points by temporal proximity and 
                metric correlation, identifying distinct evolutionary phases in tennis 
                development patterns.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Footer with methodological notes
        st.markdown("---")
        st.markdown(
            """
        <div class="highlight-section">
            <h6>Statistical Methodology</h6>
            <p><strong>Data Aggregation:</strong> Yearly means with minimum sample size requirements<br>
            <strong>Trend Analysis:</strong> Linear regression with R² significance testing (p < 0.05)<br>
            <strong>Change Detection:</strong> Piecewise regression with 5% improvement thresholds<br>
            <strong>Phase Clustering:</strong> Temporal proximity analysis with 3-year windows</p>
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
