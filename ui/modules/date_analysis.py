"""
Tennis Analysis - Date Analysis Page
Interactive temporal pattern analysis and date visualization.
"""

import pandas as pd
import streamlit as st


def render_date_analysis_page():
    """Render the complete date analysis page."""

    try:
        # Page header
        st.markdown("# Date & Temporal Analysis")
        st.markdown("### Statistical Analysis of Tennis Match Temporal Patterns")
        st.markdown("---")

        # Introduction
        st.markdown("## Overview")
        st.markdown("""
        Comprehensive temporal analysis of professional tennis match data, examining seasonal patterns, 
        yearly trends, and match distribution across different time scales. Statistical methods identify 
        peak activity periods, seasonal variations, and long-term evolutionary patterns in tennis scheduling.
        
        **Analysis Components:**
        
        - **Annual Trends** - Year-over-year match volume analysis with trend identification
        - **Seasonal Patterns** - Monthly and quarterly distribution analysis
        - **Weekly Cycles** - Tournament scheduling patterns throughout the calendar year
        - **Match Intensity** - Daily match concentration and scheduling density analysis
        """)

        # Import components with error handling
        try:
            from ui.components.chart_utils import create_plotly_chart
            from ui.components.date_analysis import (
                create_annual_matches_chart,
                create_era_timeline_chart,
                create_match_intensity_chart,
                create_monthly_timeline_chart,
                create_quarterly_heatmap,
                create_seasonal_patterns_chart,
                create_surface_timeline_chart,
                create_weekly_patterns_chart,
                get_date_analysis_data,
                get_temporal_statistics,
            )

            component_import_success = True
        except Exception as e:
            st.error(f"Error importing date analysis components: {e}")
            component_import_success = False

        if not component_import_success:
            st.stop()

        # Load data with progress indicators
        data_placeholder = st.empty()
        with data_placeholder:
            with st.spinner("Loading temporal analysis data..."):
                df = get_date_analysis_data()

                if df is None:
                    st.error("Unable to load date analysis data")
                    st.stop()

                st.success(f"✅ Loaded {len(df):,} matches for analysis")

                # Show data info
                if len(df) < 100000:
                    st.info("📊 Using full dataset")
                else:
                    st.info("🎯 Using optimized sample for fast visualization")

        # Calculate statistics
        with st.spinner("Calculating temporal statistics..."):
            stats = get_temporal_statistics(df)

        # Clear loading messages
        data_placeholder.empty()

        # Check if we have valid statistics
        if not stats:
            st.error("Unable to calculate temporal statistics")
            st.stop()

        # Summary metrics
        st.markdown("## Temporal Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Matches", f"{stats.get('total_matches', 0):,}")

        with col2:
            if "date_range" in stats and stats["date_range"]:
                date_range = stats["date_range"]
                range_str = f"{date_range[0].strftime('%Y')} - {date_range[1].strftime('%Y')}"
                st.metric("Date Range", range_str)
            else:
                st.metric("Date Range", "Unknown")

        with col3:
            st.metric("Years Covered", f"{stats.get('years_covered', 0)}")

        with col4:
            st.metric("Avg Matches/Year", f"{stats.get('avg_matches_per_year', 0):.0f}")

        # Additional insights
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            busiest_year = stats.get("busiest_year", "Unknown")
            busiest_count = stats.get("busiest_year_count", 0)
            st.metric("Busiest Year", f"{busiest_year}", f"{busiest_count:,} matches")

        with col6:
            busiest_month = stats.get("busiest_month", "Unknown")
            busiest_month_count = stats.get("busiest_month_count", 0)
            st.metric("Busiest Month", busiest_month, f"{busiest_month_count:,} matches")

        with col7:
            seasonal_var = stats.get("seasonal_variation", 0)
            st.metric("Seasonal Variation", f"{seasonal_var:.0f}%")

        with col8:
            avg_monthly = stats.get("avg_matches_per_month", 0)
            st.metric("Avg Matches/Month", f"{avg_monthly:.0f}")

        # Navigation tabs with lazy loading
        tab1, tab2, tab3, tab4 = st.tabs(["Core Patterns", "Advanced Analysis", "Surface & Era Trends", "Statistical Insights"])

        with tab1:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)

            # Core temporal visualizations
            st.markdown("## Core Temporal Patterns")

            col1, col2 = st.columns(2)

            with col1:
                with st.spinner("Creating annual chart..."):
                    try:
                        fig_annual = create_annual_matches_chart(df)
                        create_plotly_chart(
                            fig_annual,
                            chart_key="annual_matches_chart",
                            caption="Interactive annual match count analysis with trend overlay",
                        )
                    except Exception as e:
                        st.error(f"Error creating annual chart: {e}")
                        import traceback

                        with st.expander("Debug Info"):
                            st.code(traceback.format_exc())

            with col2:
                with st.spinner("Creating seasonal chart..."):
                    try:
                        fig_seasonal = create_seasonal_patterns_chart(df)
                        create_plotly_chart(
                            fig_seasonal,
                            chart_key="seasonal_patterns_chart",
                            caption="Seasonal distribution showing peak tournament months",
                        )
                    except Exception as e:
                        st.error(f"Error creating seasonal chart: {e}")
                        import traceback

                        with st.expander("Debug Info"):
                            st.code(traceback.format_exc())

            # Monthly timeline
            with st.spinner("Creating monthly timeline..."):
                try:
                    fig_monthly = create_monthly_timeline_chart(df)
                    create_plotly_chart(
                        fig_monthly,
                        chart_key="monthly_timeline_chart",
                        caption="Monthly match distribution timeline showing long-term patterns",
                    )
                except Exception as e:
                    st.error(f"Error creating monthly timeline: {e}")
                    import traceback

                    with st.expander("Debug Info"):
                        st.code(traceback.format_exc())

            st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)

            # Advanced temporal analysis
            st.markdown("## Advanced Temporal Analysis")

            col1, col2 = st.columns(2)

            with col1:
                try:
                    fig_quarterly = create_quarterly_heatmap(df)
                    create_plotly_chart(
                        fig_quarterly,
                        chart_key="quarterly_heatmap",
                        caption="Quarterly distribution heatmap revealing seasonal intensity patterns",
                    )
                except Exception as e:
                    st.error(f"Error creating quarterly heatmap: {e}")

            with col2:
                try:
                    fig_intensity = create_match_intensity_chart(df)
                    create_plotly_chart(
                        fig_intensity,
                        chart_key="match_intensity_chart",
                        caption="Match intensity distribution showing daily scheduling patterns",
                    )
                except Exception as e:
                    st.error(f"Error creating intensity chart: {e}")

            # Weekly patterns
            try:
                fig_weekly = create_weekly_patterns_chart(df)
                create_plotly_chart(
                    fig_weekly, chart_key="weekly_patterns_chart", caption="Weekly distribution throughout the year with seasonal markers"
                )
            except Exception as e:
                st.error(f"Error creating weekly patterns: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)

            # Surface and era analysis
            st.markdown("## Surface & Era Temporal Trends")

            try:
                fig_surface = create_surface_timeline_chart(df)
                create_plotly_chart(
                    fig_surface,
                    chart_key="surface_timeline_chart",
                    caption="Surface distribution evolution over time showing preference changes",
                )
            except Exception as e:
                st.error(f"Error creating surface timeline: {e}")

            # Era timeline (if era data is available)
            try:
                fig_era = create_era_timeline_chart(df)
                if fig_era is not None:
                    create_plotly_chart(
                        fig_era, chart_key="era_timeline_chart", caption="Era distribution over time showing tennis evolution phases"
                    )
                else:
                    st.info("Era timeline not available - requires era classification in dataset")
            except Exception as e:
                st.error(f"Error creating era timeline: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)

            # Statistical insights
            st.markdown("## Statistical Insights")

            # Temporal pattern analysis
            st.markdown("### 📊 Temporal Pattern Analysis")

            insight_col1, insight_col2, insight_col3 = st.columns(3)

            with insight_col1:
                st.markdown(
                    f"""
                <div class="insight-card">
                    <h5>Annual Growth</h5>
                    <p><strong>Peak Activity:</strong> {stats['busiest_year']}<br>
                    <strong>Growth Period:</strong> {stats['years_covered']} years<br>
                    <strong>Activity Level:</strong> {stats['avg_matches_per_year']:.0f} matches/year</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with insight_col2:
                st.markdown(
                    f"""
                <div class="insight-card">
                    <h5>Seasonal Patterns</h5>
                    <p><strong>Peak Month:</strong> {stats['busiest_month']}<br>
                    <strong>Quiet Month:</strong> {stats['quietest_month']}<br>
                    <strong>Variation:</strong> {stats['seasonal_variation']:.0f}% difference</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with insight_col3:
                surface_top = max(stats["surface_distribution"], key=stats["surface_distribution"].get)
                surface_pct = stats["surface_distribution"][surface_top] / stats["total_matches"] * 100
                st.markdown(
                    f"""
                <div class="insight-card">
                    <h5>Surface Preference</h5>
                    <p><strong>Dominant Surface:</strong> {surface_top}<br>
                    <strong>Share:</strong> {surface_pct:.1f}%<br>
                    <strong>Distribution:</strong> {len(stats['surface_distribution'])} surface types</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Coverage analysis
            st.markdown("### 🔍 Data Coverage Analysis")

            # Calculate coverage metrics
            all_months_in_range = (stats["date_range"][1].year - stats["date_range"][0].year + 1) * 12
            actual_months = df["month_period"].nunique()
            coverage_pct = actual_months / all_months_in_range * 100

            coverage_col1, coverage_col2 = st.columns(2)

            with coverage_col1:
                st.markdown(
                    f"""
                **Expected Coverage:**
                - Expected months in range: {all_months_in_range}
                - Months with data: {actual_months}
                - Coverage percentage: {coverage_pct:.1f}%
                """
                )

            with coverage_col2:
                # Find gaps if any
                all_months = set(df["month_period"])
                date_range = stats["date_range"]
                expected_months = set(
                    pd.period_range(
                        start=f"{date_range[0].year}-{date_range[0].month:02d}",
                        end=f"{date_range[1].year}-{date_range[1].month:02d}",
                        freq="M",
                    )
                )
                missing_months = expected_months - all_months

                if missing_months:
                    missing_count = len(missing_months)
                    st.markdown(f"**Data Gaps:** {missing_count} missing months detected")
                else:
                    st.markdown("**Data Quality:** Complete temporal coverage ✅")

            st.markdown("</div>", unsafe_allow_html=True)

        # Footer with methodological notes
        st.markdown("---")
        st.markdown(
            """
        <div class="highlight-section">
            <h6>Temporal Analysis Methodology</h6>
            <p><strong>Data Processing:</strong> Date validation with timezone normalization<br>
            <strong>Temporal Aggregation:</strong> Multiple time scales (daily, weekly, monthly, quarterly, yearly)<br>
            <strong>Pattern Detection:</strong> Seasonal decomposition with trend analysis<br>
            <strong>Statistical Validation:</strong> Coverage analysis and data quality assessment</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Critical error in Date Analysis page: {e}")
        import traceback

        st.code(traceback.format_exc())


if __name__ == "__main__":
    render_date_analysis_page()
