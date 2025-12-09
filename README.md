# Crop Production & Yield Analysis Dashboard
    
This is a Streamlit-based interactive dashboard for analyzing crop production and yield trends in India. It provides actionable insights for policymakers, researchers, and agriculture enthusiasts by visualizing key metrics and identifying critical trends.

## Features

- **Interactive Filtering**: Dynamic filtering by State, District, Crop, Season, and Year range allows for granular analysis.
- **Key Metrics Overview**: Real-time calculation of Total Area (ha), Total Production (tonnes), and Average Yield (tonnes/ha).
- **Advanced Visualizations**:
    - **Trend Analysis**: Multi-year linear charts showing trends in Area, Production, and Yield.
    - **Geospatial Insights**: Interactive Choropleth maps visualizing yield distribution across Indian states.
    - **Correlation Analysis**: Heatmaps to explore relationships between Area, Production, and Yield.
    - **Scatter Plots**: Bubble charts visualizing Yield vs. Area (sized by Production) to identify high-performing regions.
    - **District-level deep dives**: Top producing districts bar charts.
- **Automated Policy Briefs**:
    - Detects states with significant yield decline (>10% threshold).
    - Auto-generates a downloadable PDF Policy Brief summarizing context, trends, and recommended actions.

## Analysis Methodology

The dashboard computes **Yield** as:
$$ Yield = \frac{\text{Total Production}}{\text{Total Area}} $$

### Yield Decline Detection
To identify regions needing attention, the system:
1. Filters data for the selected crop and year range.
2. Compares the weighted average yield of the start year vs. the end year for each state.
3. Flags states where the percentage change is $\le -10\%$.

## Data Sources

The dashboard relies on the following datasets:
- `India Agriculture Crop Production.csv`: Historical data on crop area and production.
- `india_state_geo.json`: GeoJSON geometry for rendering Indian state maps.

## Installation & Usage

1. **Prerequisites**
   Ensure you have Python installed.

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**
   ```bash
   streamlit run app.py
   ```
