"""
Tennis Analysis - Era Analysis Page
Interactive analysis of tennis performance across different eras.
"""

import streamlit as st


def render_era_analysis_page():
    """Render the complete era analysis page."""

    try:
        # Page header
        st.markdown("# Era Analysis")
        st.markdown("### Statistical Analysis of Tennis Performance Evolution")
        st.markdown("---")

        # Introduction
        st.markdown("## Overview")
        st.markdown("""
        Statistical analysis of professional tennis evolution across distinct temporal and stylistic eras. 
        Performance metrics, playing styles, and surface preferences are analyzed using standardized 
        statistical methods to identify significant trends and era-defining characteristics.
        
        **Era Classification:**
        
        - **Classic Era** - Traditional serve-and-volley and baseline play patterns
        - **Transition Era** - Evolution towards power baseline strategies  
        - **Modern Era** - Athletic baseline dominance with increased court coverage
        - **Current Era** - Diverse playing styles with ultra-athletic performance standards
        """)

        # Import components with error handling
        try:
            from ui.components.era_analysis import display_era_champions, display_era_overview, display_era_trends, display_surface_analysis

            component_import_success = True
        except Exception as e:
            st.error(f"Error importing era analysis components: {e}")
            component_import_success = False

        if not component_import_success:
            st.stop()

        # Navigation tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Surface Analysis", "Champions"])

        with tab1:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
            try:
                with st.container():
                    display_era_overview()
            except Exception as e:
                st.error(f"Error in Era Overview: {e}")
                import traceback

                st.code(traceback.format_exc())
            st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
            try:
                with st.container():
                    display_era_trends()
            except Exception as e:
                st.error(f"Error in Era Trends: {e}")
                import traceback

                st.code(traceback.format_exc())
            st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
            try:
                with st.container():
                    display_surface_analysis()
            except Exception as e:
                st.error(f"Error in Surface Analysis: {e}")
                import traceback

                st.code(traceback.format_exc())
            st.markdown("</div>", unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
            try:
                with st.container():
                    display_era_champions()
            except Exception as e:
                st.error(f"Error in Era Champions: {e}")
                import traceback

                st.code(traceback.format_exc())
            st.markdown("</div>", unsafe_allow_html=True)

        # Research findings section
        st.markdown("---")
        st.markdown("## Key Findings")

        insight_col1, insight_col2, insight_col3 = st.columns(3)

        with insight_col1:
            st.markdown(
                """
            <div class="insight-card">
                <h5>Performance Evolution</h5>
                <p>Quantitative analysis reveals systematic increases in service dominance and 
                athletic performance metrics across eras, with statistical significance testing 
                confirming evolutionary trends.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with insight_col2:
            st.markdown(
                """
            <div class="insight-card">
                <h5>Statistical Trends</h5>
                <p>Service metrics show positive correlation with era progression, while defensive 
                capabilities demonstrate corresponding improvements, resulting in longer rallies 
                and strategic complexity.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with insight_col3:
            st.markdown(
                """
            <div class="insight-card">
                <h5>Surface Differentiation</h5>
                <p>Playing style preferences demonstrate significant surface-specific variations, 
                with hard courts favoring power metrics and clay courts emphasizing endurance 
                and consistency statistics.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Footer with methodological notes
        st.markdown("---")
        st.markdown(
            """
        <div class="highlight-section">
            <h6>Methodology</h6>
            <p><strong>Data Source:</strong> ATP Tour official match data (2005-2024)<br>
            <strong>Era Classification:</strong> Temporal and stylistic pattern analysis<br>
            <strong>Statistical Methods:</strong> Z-score normalization for cross-era comparison<br>
            <strong>Sample Requirements:</strong> Minimum thresholds enforced for statistical significance</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Critical error in Era Analysis page: {e}")
        import traceback

        st.code(traceback.format_exc())
