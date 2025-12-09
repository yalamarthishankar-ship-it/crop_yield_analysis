# Crop Production & Yield Analysis Dashboard

This is a Streamlit-based interactive dashboard for analyzing crop production and yield trends in India.

## Features

- **Interactive Dashboard**: Filter data by State, District, Crop, Season, and Year.
- **Visualizations**:
    - Time series analysis of Area, Production, and Yield.
    - Interactive Maps (Choropleth) showing yield distribution across states.
    - Correlation matrices and scatter plots (Yield vs Area).
    - Crop-wise production trends.
- **Policy Brief Generation**: Automatically identifying states with significant yield decline and generating a downloadable PDF policy brief.

## Data Sources

The dashboard uses:
- `India Agriculture Crop Production.csv`
- `india_state_geo.json` (GeoJSON for map visualizations)

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```
