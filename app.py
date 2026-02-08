"""
Seattle Edge Accessibility Dashboard
Government dashboard for monitoring sidewalk/accessibility issues across Seattle edges.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from pathlib import Path

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Seattle Edge Accessibility Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data path
DATA_PATH = Path(__file__).parent / "gnn_input.csv"

# Label options for Issue Metrics (without _avg suffix for display)
LABEL_OPTIONS = {
    "CurbRamp": "CurbRamp_avg",
    "NoCurbRamp": "NoCurbRamp_avg",
    "NoSidewalk": "NoSidewalk_avg",
    "Obstacle": "Obstacle_avg",
    "Occlusion": "Occlusion_avg",
    "Other": "Other_avg",
    "SurfaceProblem": "SurfaceProblem_avg",
}


def load_data():
    """Load and preprocess the GNN input data."""
    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        return None
    
    df = pd.read_csv(DATA_PATH)
    
    # Handle first column (Unnamed: 0 or index)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    
    # Ensure required columns exist
    required = ["severitynum_barriers", "u_lat", "u_lon", "v_lat", "v_lon"]
    for col in required:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return None
    
    return df


def hex_to_rgba(hex_color: str, alpha: int = 255) -> list:
    """Convert hex color to RGBA list for PyDeck."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return [r, g, b, alpha]


def tableau_gradient(value: float, min_val: float, max_val: float) -> list:
    """
    Map a value to a Tableau-style sequential gradient (Blue -> Green -> Yellow -> Red).
    Low values = cool blue, High values = hot red.
    """
    if max_val == min_val:
        return hex_to_rgba("#4A90D9", 180)  # Default blue
    
    # Normalize to 0-1
    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0, min(1, normalized))
    
    # Tableau sequential: Blue (#4477AA) -> Cyan -> Green -> Yellow -> Red (#EE3333)
    # Simplified 5-stop gradient
    stops = [
        (0.0, "#2E86AB"),    # Dark blue - low
        (0.25, "#4ECDC4"),   # Teal
        (0.5, "#95E1A3"),    # Light green
        (0.75, "#F7DC6F"),   # Yellow
        (1.0, "#E74C3C"),    # Red - high
    ]
    
    for i in range(len(stops) - 1):
        if stops[i][0] <= normalized <= stops[i + 1][0]:
            t = (normalized - stops[i][0]) / (stops[i + 1][0] - stops[i][0])
            # Interpolate between the two colors
            c1 = stops[i][1]
            c2 = stops[i + 1][1]
            r = int(int(c1[1:3], 16) * (1 - t) + int(c2[1:3], 16) * t)
            g = int(int(c1[3:5], 16) * (1 - t) + int(c2[3:5], 16) * t)
            b = int(int(c1[5:7], 16) * (1 - t) + int(c2[5:7], 16) * t)
            return [r, g, b, 180]
    
    return hex_to_rgba(stops[-1][1], 180)


def create_line_layer_df(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Create DataFrame for PyDeck with path and color for each edge.
    """
    valid = df[["u_lat", "u_lon", "v_lat", "v_lon", value_col]].dropna()
    
    if valid.empty:
        return pd.DataFrame()
    
    min_val = valid[value_col].min()
    max_val = valid[value_col].max()
    
    result = []
    for _, row in valid.iterrows():
        color = tableau_gradient(row[value_col], min_val, max_val)
        result.append({
            "path": [[row["u_lon"], row["u_lat"]], [row["v_lon"], row["v_lat"]]],
            "color": color,
            "value": row[value_col]
        })
    
    return pd.DataFrame(result)


def render_map(df_map: pd.DataFrame, value_col: str, title: str, show_legend: bool = True):
    """Render a PyDeck map with edges."""
    if df_map.empty:
        st.warning("No data to display on the map.")
        return
    
    # Seattle center - compute from path coordinates
    try:
        def get_center(path):
            if path and len(path) >= 2:
                return (path[0][1] + path[1][1]) / 2, (path[0][0] + path[1][0]) / 2
            return 47.6062, -122.3321
        centers = df_map["path"].apply(get_center)
        lat_center = centers.apply(lambda x: x[0]).mean()
        lon_center = centers.apply(lambda x: x[1]).mean()
    except Exception:
        lat_center, lon_center = 47.6062, -122.3321
    
    # PyDeck PathLayer - edges as lines with gradient coloring
    layer = pdk.Layer(
        "PathLayer",
        df_map,
        get_path="path",
        get_color="color",
        get_width=3,
        width_scale=10,
        width_min_pixels=1.5,
        width_max_pixels=5,
        pickable=True,
        auto_highlight=True,
    )
    
    view_state = pdk.ViewState(
        latitude=lat_center,
        longitude=lon_center,
        zoom=10,
        pitch=0,
        bearing=0
    )
    
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip={
            "html": "<b>Value:</b> {value}",
            "style": {"backgroundColor": "steelblue", "color": "white", "padding": "8px"}
        }
    )
    
    st.pydeck_chart(r, use_container_width=True)
    
    if show_legend:
        min_val = df_map["value"].min()
        max_val = df_map["value"].max()
        # Tableau-style gradient legend (Blue → Teal → Green → Yellow → Red)
        st.markdown(f"""
        <div style="margin-top: 12px; padding: 10px; background: #f8f9fa; border-radius: 8px;">
            <div style="font-weight: 600; margin-bottom: 6px;">Gradient scale ({value_col})</div>
            <div style="height: 12px; background: linear-gradient(to right, #2E86AB, #4ECDC4, #95E1A3, #F7DC6F, #E74C3C); border-radius: 4px; margin-bottom: 4px;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; color: #555;">
                <span>{min_val:.1f} (low)</span>
                <span>{max_val:.1f} (high)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def home_tab(df: pd.DataFrame):
    """Home tab - Seattle map with edges color-coded by num_barriers."""
    st.header("🗺️ Seattle Edge Accessibility Overview")
    st.markdown("""
    This map shows all sidewalk/accessibility edges across Seattle. 
    **Edges are color-coded by the number of barriers** (severitynum_barriers) using a gradient scale—cooler colors indicate fewer issues, warmer colors indicate more problems.
    """)
    
    # Optional: filter edges with barriers for clearer view, or show all
    col1, col2 = st.columns([1, 4])
    with col1:
        show_zero = st.checkbox("Include edges with 0 barriers", value=True, key="home_show_zero")
    
    if not show_zero:
        df_filtered = df[df["severitynum_barriers"] > 0]
    else:
        df_filtered = df
    
    # Sample if too many for performance (pydeck can handle ~50k but may be slow)
    max_edges = 25000
    if len(df_filtered) > max_edges:
        st.info(f"Showing sample of {max_edges:,} edges for performance. Use filters to narrow down.")
        df_filtered = df_filtered.sample(n=max_edges, random_state=42)
    
    df_edges = create_line_layer_df(df_filtered, "severitynum_barriers")
    render_map(df_edges, "severitynum_barriers", "Barrier Count", show_legend=True)
    
    # Summary stats
    st.subheader("Summary Statistics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Edges", f"{len(df):,}")
    with c2:
        edges_with_issues = (df["severitynum_barriers"] > 0).sum()
        st.metric("Edges with Barriers", f"{edges_with_issues:,}")
    with c3:
        st.metric("Max Barriers (single edge)", f"{df['severitynum_barriers'].max():.0f}")
    with c4:
        st.metric("Avg Barriers (edges with issues)", f"{df[df['severitynum_barriers']>0]['severitynum_barriers'].mean():.2f}")


def issue_metrics_tab(df: pd.DataFrame):
    """Issue Metrics tab - Select label and see map color-coded by label_avg (excluding zeros)."""
    st.header("📊 Issue Metrics by Label")
    st.markdown("""
    Select an issue type below to see where those specific problems exist across Seattle. 
    The map is color-coded by the **average severity** (label_avg) for that issue type.
    **Only edges with non-zero values** for the selected label are shown.
    """)
    
    selected_label = st.selectbox(
        "Select Issue Type",
        options=list(LABEL_OPTIONS.keys()),
        key="issue_label_select"
    )
    
    avg_col = LABEL_OPTIONS[selected_label]
    
    # Filter out rows where this label's avg is 0
    df_filtered = df[df[avg_col] > 0].copy()
    
    if df_filtered.empty:
        st.warning(f"No edges have {selected_label} issues (all values are 0).")
        return
    
    # Sample for performance if needed
    max_edges = 20000
    if len(df_filtered) > max_edges:
        st.info(f"Showing sample of {max_edges:,} edges with {selected_label} issues.")
        df_filtered = df_filtered.sample(n=max_edges, random_state=42)
    
    df_edges = create_line_layer_df(df_filtered, avg_col)
    render_map(df_edges, avg_col, f"{selected_label} Average Severity", show_legend=True)
    
    # Stats for this label
    st.subheader(f"{selected_label} Statistics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Edges with this issue", f"{len(df_filtered):,}")
    with c2:
        st.metric("Average severity (mean)", f"{df_filtered[avg_col].mean():.2f}")
    with c3:
        st.metric("Max severity", f"{df_filtered[avg_col].max():.2f}")


def placeholder_viz_tab(name: str, description: str):
    """Placeholder for future visualizations."""
    st.header(f"📈 {name}")
    st.info(f"**Coming soon:** {description}")
    st.markdown("---")
    st.markdown("_Add your visualization requirements and we'll implement them here._")


def main():
    # Custom CSS for navigation bar
    st.markdown("""
    <style>
    .nav-bar {
        display: flex;
        gap: 1rem;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    .nav-bar a {
        text-decoration: none;
        color: #1f77b4;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        transition: background 0.2s;
    }
    .nav-bar a:hover {
        background: #e3f2fd;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🗺️ Seattle Edge Accessibility Dashboard")
    st.markdown("**Government dashboard for monitoring sidewalk and accessibility issues across Seattle**")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home",
        "📊 Issue Metrics",
        "📈 Visualization 3",
        "📉 Visualization 4",
        "📋 Visualization 5"
    ])
    
    with tab1:
        home_tab(df)
    
    with tab2:
        issue_metrics_tab(df)
    
    with tab3:
        placeholder_viz_tab(
            "Visualization 3",
            "Add your third interactive visualization here."
        )
    
    with tab4:
        placeholder_viz_tab(
            "Visualization 4",
            "Add your fourth interactive visualization here."
        )
    
    with tab5:
        placeholder_viz_tab(
            "Visualization 5",
            "Add your fifth interactive visualization here."
        )


if __name__ == "__main__":
    main()
