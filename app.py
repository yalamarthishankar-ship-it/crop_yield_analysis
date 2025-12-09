import io
import json
import datetime
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import geopandas as gd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


# Configurations
st.set_page_config(
    page_title="Crop Production & Yield Dashboard",
    layout="wide",
)

# Data Sources
CROP_DATA_PATH = r"Inputs\India Agriculture Crop Production.csv"
MAPS_DATA_PATH = r"Inputs\india_state_geo.json"

# Data Loading Helpers
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """
    Loads and preprocesses crop production data from a CSV file.

    Parameters:
        path (str): The file path to the CSV dataset.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the clean crop data with standardized column names
                      and numeric conversions for key metrics (Year, Area, Production, Yield).

    Raises:
        ValueError: If required columns are missing from the CSV.
    """
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = [c.strip().title() for c in df.columns]

    required_cols = ["State", "District", "Crop", "Season", "Year", "Area", "Production", "Yield"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Year Parsing
    df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})").astype(float).astype("Int64")

    # Ensure numeric
    for col in ["Year", "Area", "Production", "Yield"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data
def load_geojson(path: str):
    """
    Loads the GeoJSON file for Indian states and standardizes state names for mapping.

    Parameters:
        path (str): The file path to the GeoJSON file.

    Returns:
        dict: A JSON object representing the map features, or None if loading fails.
              State names are harmonized to match the crop dataset (e.g., 'Orissa' -> 'Odisha').
    """
    try:
        gj = gd.read_file(MAPS_DATA_PATH)
        
        gj = gj[['NAME_1', 'geometry']].rename(
            columns={
                'NAME_1': 'State'
            }
        )
        
        state_mapping = {
            'Andaman and Nicobar': 'Andaman and Nicobar Islands',
            'Orissa': 'Odisha',
            'Uttaranchal': 'Uttarakhand'
        }
        
        gj['State'] = gj['State'].replace(state_mapping)
        
        return json.loads(gj.to_json())
    except Exception:
        return None
    

# Analytical Helpers
def filter_data(
    df: pd.DataFrame,
    states: List[str],
    crops: List[str],
    districts: List[str],
    seasons: List[str],
    years: List[int],
) -> pd.DataFrame:
    """
    Filters the DataFrame based on the provided criteria.
    Parameters:
        df (pd.DataFrame): The input DataFrame to filter.
        states (List[str]): List of states to filter by.
        crops (List[str]): List of crops to filter by.
        districts (List[str]): List of districts to filter by.
        seasons (List[str]): List of seasons to filter by.
        years (List[int]): List of years to filter by.
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    mask = pd.Series(True, index=df.index)

    if states:
        mask &= df["State"].isin(states)
    if crops:
        mask &= df["Crop"].isin(crops)
    if districts:
        mask &= df["District"].isin(districts)
    if seasons:
        mask &= df["Season"].isin(seasons)
    if years:
        mask &= df["Year"].isin(years)

    return df[mask].copy()


def make_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the DataFrame by Year, summing Area and Production, and calculating weighted Yield.
    Parameters:
        df (pd.DataFrame): The input DataFrame to aggregate.
    Returns:
        pd.DataFrame: The aggregated DataFrame with columns Year, Area, Production, and Yield.
    """
    if df.empty:
        return df

    agg = (
        df.groupby("Year")
        .agg(
            Area=("Area", "sum"),
            Production=("Production", "sum"),
        )
        .reset_index()
    )
    # Weighted yield: total production / total area
    agg["Yield"] = agg["Production"] / agg["Area"].replace({0: np.nan})
    return agg


def states_with_yield_decline(
    df: pd.DataFrame, crop: str, years: List[int], threshold: float = -10.0
) -> pd.DataFrame:
    """
    For the selected crop + year range, compute % change in yield by state.
    Returns states where change <= threshold (e.g., -10%).
    """
    if df.empty or len(years) < 2:
        return pd.DataFrame()

    yrs_sorted = sorted(years)
    start_year, end_year = yrs_sorted[0], yrs_sorted[-1]

    cdf = df[df["Crop"] == crop].copy()
    if cdf.empty:
        return pd.DataFrame()

    def agg_year(y: int) -> pd.Series:
        tmp = cdf[cdf["Year"] == y].copy()
        if tmp.empty:
            return pd.Series(dtype=float)
        grouped = (
            tmp.groupby("State")
            .apply(lambda g: g["Production"].sum() / g["Area"].sum() if g["Area"].sum() > 0 else np.nan)
            .rename("Yield")
        )
        grouped = grouped.to_frame()
        grouped["Year"] = y
        return grouped

    start = agg_year(start_year)
    end = agg_year(end_year)

    if start.empty or end.empty:
        return pd.DataFrame()

    merged = (
        start[["Yield"]]
        .rename(columns={"Yield": "Yield_Start"})
        .join(end[["Yield"]].rename(columns={"Yield": "Yield_End"}), how="inner")
    )

    merged["Pct_Change_Yield"] = (merged["Yield_End"] - merged["Yield_Start"]) / merged["Yield_Start"] * 100
    declined = merged[merged["Pct_Change_Yield"] <= threshold].reset_index()
    declined = declined.rename(columns={"index": "State"})

    return declined.sort_values("Pct_Change_Yield")


def build_policy_brief(
    df_filtered: pd.DataFrame,
    ts_df: pd.DataFrame,
    decline_df: pd.DataFrame,
    selected_states: List[str],
    selected_crops: List[str],
    selected_seasons: List[str],
    selected_years: List[int],
) -> str:
    """
    Generates a text-based policy brief summarizing the analysis results.

    The brief includes:
    1. Context (selected filters).
    2. Aggregate indicators (Total Area, Production, Yield).
    3. Yield trend overview.
    4. Top producing states.
    5. List of states with significant yield decline.
    6. Recommended follow-up actions.

    Parameters:
        df_filtered (pd.DataFrame): The filtered dataset.
        ts_df (pd.DataFrame): Time-series data for the selected context.
        decline_df (pd.DataFrame): DataFrame containing states with yield decline.
        selected_states (List[str]): List of selected states.
        selected_crops (List[str]): List of selected crops.
        selected_seasons (List[str]): List of selected seasons.
        selected_years (List[int]): List of selected years.

    Returns:
        str: A multi-line string containing the formatted policy brief.
    """
    if df_filtered.empty:
        return "No data available for the selected filters to generate a policy brief."

    years_str = f"{min(selected_years)}–{max(selected_years)}" if selected_years else "N/A"
    crops_str = ", ".join(selected_crops) if selected_crops else "All crops"
    states_str = ", ".join(selected_states) if selected_states else "All states"
    seasons_str = ", ".join(selected_seasons) if selected_seasons else "All seasons"

    total_area = df_filtered["Area"].sum()
    total_prod = df_filtered["Production"].sum()
    overall_yield = total_prod / total_area if total_area > 0 else np.nan

    # Trend sentence
    trend_sentence = ""
    if not ts_df.empty and len(ts_df) >= 2:
        first_y, last_y = ts_df.iloc[0], ts_df.iloc[-1]
        pct_yield_change = (last_y["Yield"] - first_y["Yield"]) / first_y["Yield"] * 100 if first_y["Yield"] else np.nan
        if pd.notna(pct_yield_change):
            direction = "increase" if pct_yield_change > 0 else "decline"
            trend_sentence = (
                f"Over {years_str}, average yield shows a {direction} of "
                f"{pct_yield_change:0.1f}% for the selected filters."
            )

    # Top states by production
    by_state = (
        df_filtered.groupby("State")
        .agg(total_prod=("Production", "sum"))
        .sort_values("total_prod", ascending=False)
        .head(3)
    )
    top_states_summary = "; ".join(
        f"{s}: {p:0.1f} tonnes" for s, p in zip(by_state.index, by_state["total_prod"])
    )

    # Decline states
    decline_lines = []
    if not decline_df.empty:
        for _, row in decline_df.iterrows():
            decline_lines.append(
                f"- {row['State']}: {row['Pct_Change_Yield']:.1f}% change in yield "
                f"(from {row['Yield_Start']:.2f} to {row['Yield_End']:.2f})"
            )

    decline_block = (
        "\n".join(decline_lines)
        if decline_lines
        else "No states show a yield decline greater than the configured threshold."
    )

    brief = f"""
Crop Production & Yield Policy Brief
====================================

1. Context
----------
- States: {states_str}
- Crops: {crops_str}
- Seasons: {seasons_str}
- Years: {years_str}

2. Aggregate Indicators
------------------------
- Total area sown: {total_area:,.2f} ha
- Total production: {total_prod:,.2f} tonnes
- Overall average yield: {overall_yield:,.2f} t/ha

3. Yield Trend Overview
-----------------------
{trend_sentence or "Insufficient data to compute a clear yield trend."}

4. Top Producing States
-----------------------
{top_states_summary or "Insufficient data."}

5. States With Significant Yield Decline
----------------------------------------
(Threshold: more than 10% decline over the selected period)

{decline_block}

6. Recommended Follow-up Actions
--------------------------------
- Investigate agronomic and climatic factors in declining states.
- Review seed, fertilizer, and irrigation support in low-yield districts.
- Target extension services and demonstrations towards declining hotspots.
- Encourage data-backed monitoring of yield trends at district level.

Generated automatically from the Crop Production & Yield Dashboard.
"""
    return brief.strip()


# UI Components
def main():
    """
    The main entry point for the Streamlit application.

    This function:
    - Sets up the dashboard layout and sidebar.
    - Loads data resources.
    - Handles user input for filtering data.
    - Renders the "Dashboard" view with metrics, charts, and maps.
    - Renders the "Map View" for state-specific deep dives.
    - Generates and allows downloading of the Policy Brief PDF.
    """
    st.title("Crop Production & Yield Dashboard")

    # Load data
    try:
        df = load_data(CROP_DATA_PATH)
        geojson = load_geojson(MAPS_DATA_PATH)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # ---- Sidebar Navigation ----
    # Initialize session state
    if "view" not in st.session_state:
        st.session_state.view = "Dashboard"
    
    # Sidebar navigation with custom button styling
    if st.sidebar.button("Dashboard", use_container_width=True, key="dashboard_btn"):
        st.session_state.view = "Dashboard"
    
    if st.sidebar.button("Map View", use_container_width=True, key="map_btn"):
        st.session_state.view = "Map View"
    
    view = st.session_state.view
    
    # ---- Collecting Dropdown Filters ----
    # Global Filter Options
    all_states = sorted(df["State"].dropna().unique().tolist())
    all_crops = sorted(df["Crop"].dropna().unique().tolist())
    all_seasons = sorted(df["Season"].dropna().unique().tolist())
    
    # Year Range: Min available year to Current Year
    min_year = int(df["Year"].min()) if not df["Year"].empty else 2000
    current_year = datetime.datetime.now().year
    all_years = list(range(min_year, current_year + 1))
    
    if view == "Dashboard":
        # ---- Filter Selection Grid ----
        grid1 = st.columns(3)
        
        with grid1[0]:
            selected_states = st.multiselect(
                "State", all_states, default="Gujarat" if "Gujarat" in all_states else all_states
            )
            selected_crops = st.multiselect(
                "Crop", all_crops, default=["Rice"] if "Rice" in all_crops else all_crops
            )
        
        with grid1[1]:
            districts_filtered = sorted(
                df[df["State"].isin(selected_states)]["District"].dropna().unique().tolist()
            )
            selected_districts = st.multiselect(
                "District", districts_filtered
            )
            selected_seasons = st.multiselect(
                "Season", all_seasons, default=["Kharif"] if "Kharif" in all_seasons else all_seasons
            )
        
        with grid1[2]:
            selected_years = st.multiselect(
                "Year", all_years, default=list(range(2016, 2022))
            )
        
        st.markdown("---")
        
        # ---- Data Filtering Based on Selections ----
        filtered_df = filter_data(
            df,
            states=selected_states,
            crops=selected_crops,
            districts=selected_districts,
            seasons=selected_seasons,
            years=selected_years,
        )

        if filtered_df.empty:
            st.warning("No data found for the selected filters. Please adjust your criteria.")
            st.stop()
        
        # ---- Top Level Summary Cards ----
        total_area = filtered_df["Area"].sum()
        total_prod = filtered_df["Production"].sum()
        avg_yield = total_prod / total_area if total_area > 0 else np.nan

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Area (ha)", f"{total_area:,.2f}")
        c2.metric("Total Production (tonnes)", f"{total_prod:,.2f}")
        c3.metric("Average Yield (t/ha)", f"{avg_yield:,.2f}")

        st.markdown("---")
        
        # ---- Correlation and Time Series Analysis ----
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Correlation Matrix (Area, Production, Yield)")
            corr_data = filtered_df[["Area", "Production", "Yield"]].dropna()
            if corr_data.empty:
                st.info("Not enough numeric data to compute correlation.")
            else:
                corr = corr_data.corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix",
                    zmin=-1,
                    zmax=1,
                    color_continuous_scale="RdYlGn",
                )
                st.plotly_chart(
                    fig_corr, use_container_width=True
                )
                # st.dataframe(corr.style.background_gradient(cmap="RdYlGn", axis=None))

        with col2:
            st.subheader("Time Series: Area, Production, Yield")
            ts_df = make_time_series(filtered_df)
            if ts_df.empty:
                st.info("No time series data.")
            else:
                ts_melt = ts_df.melt(
                    id_vars="Year", value_vars=["Area", "Production", "Yield"],
                    var_name="Metric", value_name="Value"
                )
                fig_ts = px.line(
                    ts_melt,
                    x="Year",
                    y="Value",
                    color="Metric",
                    markers=True,
                    title="Multi-year Trend",
                )
                st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("---")
        
        # ---- Top Producing Districts and Yield Map ----
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Top Producing Districts")
            by_dist = (
            filtered_df.groupby("District")
            .agg(Production=("Production", "sum"))
            .sort_values("Production", ascending=False)
            .head(10)
            .reset_index()
            )
            if by_dist.empty:
                st.info("No district-level data.")
            else:
                fig_bar = px.bar(
                    by_dist,
                    y="District",
                    x="Production",
                    orientation="h",
                    color="Production",
                    color_continuous_scale="RdYlGn",
                    title="Top 10 Producing Districts",
                )
            fig_bar.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col4:
            st.subheader("Map: Yield (State Level)")
            if geojson is None:
                st.info("GeoJSON not found / failed to load. Showing textual summary instead.")
                by_state_yield = (
                    filtered_df.groupby("State")
                    .apply(
                        lambda g: (
                            g["Production"].sum() / g["Area"].sum() if g["Area"].sum() > 0 else np.nan
                        )
                    )
                    .rename("Yield")
                    .reset_index()
                )
                st.dataframe(by_state_yield)
            else:
                by_state_yield = (
                    filtered_df.groupby("State")
                    .agg(
                        Area=("Area", "sum"),
                        Production=("Production", "sum"),
                    )
                    .reset_index()
                )
                by_state_yield["Yield"] = (
                    by_state_yield["Production"] / by_state_yield["Area"].replace({0: np.nan})
                )

                fig_map = px.choropleth(
                    by_state_yield,
                    geojson=geojson,
                    locations="State",
                    featureidkey="properties.State",
                    color="Yield",
                    color_continuous_scale="YlGn",
                    hover_name="State",
                    hover_data={"Area": True, "Production": True, "Yield": True},
                    title="Average Yield by State",
                )
                fig_map.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("---")
        
        # ---- Scatter Plot and Crop-wise Time Series ----
        col5, col6 = st.columns(2)

        with col5:
            st.subheader("Yield vs. Production Area")
            scatter_df = (
                filtered_df.groupby(["State", "District"])
                .agg(
                    Area=("Area", "sum"),
                    Production=("Production", "sum"),
                )
                .reset_index()
            )
            scatter_df["Yield"] = scatter_df["Production"] / scatter_df["Area"].replace({0: np.nan})
            if scatter_df.empty:
                st.info("No data for scatter plot.")
            else:
                fig_scatter = px.scatter(
                    scatter_df,
                    x="Area",
                    y="Yield",
                    size="Production",
                    color="State",
                    hover_name="District",
                    title="Yield vs. Area (Bubble size: Production)",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        with col6:
            st.subheader("Crop-wise Production Over Time")
            crop_ts = (
                filtered_df.groupby(["Year", "Crop"])
                .agg(Production=("Production", "sum"))
                .reset_index()
            )
            if crop_ts.empty:
                st.info("No crop-wise production data.")
            else:
                fig_crop_ts = px.line(
                    crop_ts,
                    x="Year",
                    y="Production",
                    color="Crop",
                    markers=True,
                    title="Crop-wise Production Time Series",
                )
                st.plotly_chart(fig_crop_ts, use_container_width=True)

        st.markdown("---")
        
        # ---- Yield decline detection + Policy brief ----
        st.subheader("States with >10% Yield Decline over Selected Period")

        primary_crop_for_trend = selected_crops[0] if selected_crops else None
        if primary_crop_for_trend and selected_years:
            decline_df = states_with_yield_decline(
                filtered_df, primary_crop_for_trend, selected_years, threshold=-10.0
            )
            if decline_df.empty:
                st.info("No states show more than 10% yield decline for the selected crop and years.")
            else:
                st.dataframe(
                    decline_df.style.background_gradient(
                        cmap="Reds", subset=["Pct_Change_Yield"]
                    )
                )
        else:
            decline_df = pd.DataFrame()
            st.info("Select at least one crop and two or more years to see yield decline analysis.")

        st.markdown("### Export Policy Brief")

        ts_df = make_time_series(filtered_df)
        brief_text = build_policy_brief(
            filtered_df,
            ts_df,
            decline_df,
            selected_states,
            selected_crops,
            selected_seasons,
            selected_years,
        )

        st.text_area("Preview", value=brief_text, height=300)
        # Generate PDF
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title and content
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            alignment=1
        )
        story.append(Paragraph("Crop Production & Yield Policy Brief", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            spaceAfter=6,
        )
        for line in brief_text.split('\n'):
            if line.strip():
                story.append(Paragraph(line, body_style))
            story.append(Spacer(1, 0.05*inch))
        
        doc.build(story)
        pdf_buffer.seek(0)
        
        st.download_button(
            "Download Policy Brief (PDF)",
            data=pdf_buffer.getvalue(),
            file_name="policy_brief.pdf",
            mime="application/pdf",
        )
    else:
        selected_state_map = st.selectbox("Select State for Map View", all_states)

        if geojson is None:
            st.info("GeoJSON not found / failed to load. Map view disabled.")
            st.stop()

        df_state = filter_data(
            df,
            [selected_state_map],
            [],
            [],
            [],
            [],
        )

        if df_state.empty:
            st.warning("No data for the selected state and filters.")
            st.stop()

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("District-wise Production (Aggregated)")
            by_dist_state = (
            df_state.groupby("District")
            .agg(Production=("Production", "sum"))
            .sort_values("Production", ascending=False)
            .head(5)
            .reset_index()
            )
            if by_dist_state.empty:
                st.info("No district data for this state.")
            else:
                fig_dist_bar = px.bar(
                    by_dist_state,
                    y="District",
                    x="Production",
                    orientation="h",
                    color="Production",
                    color_continuous_scale="RdYlGn",
                    title=f"Top 5 Districts in {selected_state_map} (Production)",
                )
            fig_dist_bar.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_dist_bar, use_container_width=True)

        with col_b:
            st.subheader("District-level Yield Time Series (Top 5 by production)")
            top5_dists = by_dist_state["District"].head(5).tolist()
            dist_ts = (
            df_state[df_state["District"].isin(top5_dists)]
            .groupby(["Year", "District"])
            .agg(
                Area=("Area", "sum"),
                Production=("Production", "sum"),
            )
            .reset_index()
            )
            dist_ts["Yield"] = dist_ts["Production"] / dist_ts["Area"].replace({0: np.nan})

            if dist_ts.empty:
                st.info("No time series data for districts.")
            else:
                fig_dist_ts = px.line(
                    dist_ts,
                    x="Year",
                    y="Yield",
                    color="District",
                    markers=True,
                    title="Yield Trend for Top Districts",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
            st.plotly_chart(fig_dist_ts, use_container_width=True)

        st.markdown("---")
        st.header("Map View: State-level Yield Trends")

        by_state_all = (
            df[df['Year'].isin(last_5_years := sorted(df['Year'].dropna().unique())[-5:])]
            .groupby("State")
            .agg(
            Area=("Area", "sum"),
            Production=("Production", "sum"),
            )
            .reset_index()
        )
        by_state_all["Yield"] = by_state_all["Production"] / by_state_all["Area"].replace({0: np.nan})

        fig_mv_map = px.choropleth(
            by_state_all,
            geojson=geojson,
            locations="State",
            featureidkey="properties.State",
            color="Yield",
            color_continuous_scale="YlGn",
            hover_name="State",
            title=f"Yield Map (Years: {min(last_5_years)}–{max(last_5_years) if last_5_years else ''})",
        )
        fig_mv_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_mv_map, use_container_width=True, height=600)  # Adjusted height for better visibility
        
if __name__ == "__main__":
    main()