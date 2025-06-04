"""
Tennis Era Analysis - Era Analysis Page
Interactive analysis of tennis performance across different eras.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import streamlit as st


def render_era_analysis_page():
    """Render the complete era analysis page."""

    try:
        # Page header
        st.markdown("# 🏟️ Era Analysis")
        st.markdown("### Comprehensive Analysis of Tennis Evolution Across Eras")
        st.markdown("---")

        # Introduction
        st.markdown("## 🎯 About Era Analysis")
        st.markdown("""
        Explore how professional tennis has evolved across distinct eras. This analysis examines performance metrics, 
        playing styles, surface preferences, and identifies the champions who defined each era of tennis history.
        
        **Tennis Eras Analyzed:**
        
        - **Classic Era** - Traditional baseline play and serve-and-volley dominance
        - **Transition Era** - Evolution towards power baseline games  
        - **Modern Era** - Athletic, powerful baseline dominance
        - **Current Era** - Ultra-athletic, diverse playing styles
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

        # Navigation tabs - removed redundant Charts tab
        tab1, tab2, tab3, tab4 = st.tabs(["🏟️ Overview", "📈 Trends", "🎾 Surface Analysis", "🏆 Champions"])

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

        # Insights section
        st.markdown("---")
        st.markdown("## 💡 Key Research Insights")

        insight_col1, insight_col2, insight_col3 = st.columns(3)

        with insight_col1:
            st.markdown(
                """
            <div class="tennis-card">
                <h5>🎾 Game Evolution</h5>
                <p>Tennis has evolved from serve-and-volley dominance to baseline power games, 
                with modern players showing increased athleticism and court coverage.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with insight_col2:
            st.markdown(
                """
            <div class="tennis-card">
                <h5>📊 Performance Metrics</h5>
                <p>Service dominance has increased over eras, while break point conversion 
                has become more challenging as defensive skills have improved.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with insight_col3:
            st.markdown(
                """
            <div class="tennis-card">
                <h5>🏟️ Surface Impact</h5>
                <p>Playing style preferences vary significantly by surface, with hard courts 
                favoring power players and clay courts rewarding endurance and consistency.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Footer with methodological notes
        st.markdown("---")
        st.markdown(
            """
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #228B22;">
            <h6>📋 Methodology Notes</h6>
            <p><strong>Data Source:</strong> ATP Tour official match data (2005-2024)<br>
            <strong>Era Classification:</strong> Based on historical tennis analysis and playing style evolution<br>
            <strong>Metrics:</strong> Performance indicators normalized for cross-era comparison<br>
            <strong>Sample Size:</strong> Minimum sample sizes enforced for statistical significance</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Critical error in Era Analysis page: {e}")
        import traceback

        st.code(traceback.format_exc())
