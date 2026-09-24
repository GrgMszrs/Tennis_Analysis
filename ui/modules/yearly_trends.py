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
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Evolution Trends", "Trend Analysis", "Evolution Phases"])

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
                st.error(f"Error in Evolution Trends: {e}")
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
        st.markdown("## Statistical Methodology & Tools")

        insight_col1, insight_col2, insight_col3 = st.columns(3)

        with insight_col1:
            st.markdown(
                """
            <div class="insight-card">
                <h5>Linear Regression Analysis</h5>
                <p><strong>Method:</strong> Ordinary Least Squares (OLS) regression<br>
                <strong>Purpose:</strong> Quantifies directional trends in tennis metrics over time<br>
                <strong>Output:</strong> Slope coefficients, R² correlation strength, and p-values for significance testing<br>
                <strong>Significance:</strong> p < 0.05 threshold ensures 95% confidence in trend direction</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with insight_col2:
            st.markdown(
                """
            <div class="insight-card">
                <h5>Piecewise Regression</h5>
                <p><strong>Method:</strong> Segmented linear models with breakpoint optimization<br>
                <strong>Purpose:</strong> Detects structural breaks where trend direction changes significantly<br>
                <strong>Algorithm:</strong> Tests all potential breakpoints, compares R² improvement between unified vs. segmented models<br>
                <strong>Validation:</strong> Requires >10% R² improvement and minimum 3-year segment lengths</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with insight_col3:
            st.markdown(
                """
            <div class="insight-card">
                <h5>Temporal Clustering</h5>
                <p><strong>Method:</strong> Proximity-based grouping of change points within 2-year windows<br>
                <strong>Purpose:</strong> Identifies synchronized transitions across multiple performance metrics<br>
                <strong>Criteria:</strong> Major transitions require ≥2 metrics changing simultaneously<br>
                <strong>Evolution Phases:</strong> Temporal periods between major transition clusters</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Footer with methodological notes
        st.markdown("---")
        st.markdown(
            """
        <div class="highlight-section">
            <h6>Technical Implementation Details</h6>
            <p><strong>Data Aggregation:</strong> Yearly means with n≥10 minimum sample size per metric per year<br>
            <strong>Normalization:</strong> Z-score standardization: (value - mean) / standard deviation<br>
            <strong>Trend Significance:</strong> Scipy linear regression with p < 0.05 significance threshold<br>
            <strong>Change Point Algorithm:</strong> Piecewise OLS with R² improvement ≥10% validation<br>
            <strong>Breakpoint Detection:</strong> Exhaustive search across all years with min 3-year segments<br>
            <strong>Phase Clustering:</strong> Temporal grouping within ±2 year windows, requiring ≥2 concurrent metrics<br>
            <strong>Acceleration Analysis:</strong> Second-order polynomial fitting for trend curvature detection</p>
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
