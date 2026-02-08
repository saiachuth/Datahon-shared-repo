"""
Seattle Edge Accessibility Dashboard
Government dashboard for monitoring sidewalk/accessibility issues across Seattle edges.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from pathlib import Path

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Seattle Accessibility Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data path
DATA_PATH = Path(__file__).parent / "gnn_input_neighbourhood.csv"

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

# Label to count column mapping (for neighbourhood ranking)
LABEL_COUNT_COLS = {
    "CurbRamp": "CurbRamp_count",
    "NoCurbRamp": "NoCurbRamp_count",
    "NoSidewalk": "NoSidewalk_count",
    "Obstacle": "Obstacle_count",
    "Occlusion": "Occlusion_count",
    "Other": "Other_count",
    "SurfaceProblem": "SurfaceProblem_count",
}

# Issue count columns for stacked bar chart
ISSUE_COUNT_COLUMNS = [
    "CurbRamp_count", "NoCurbRamp_count", "NoSidewalk_count",
    "Obstacle_count", "Occlusion_count", "Other_count", "SurfaceProblem_count",
]


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
    
    # Filter out invalid 'nhood' values globally
    if "nhood" in df.columns:
        # Convert to string, strip whitespace, lowercase, and filter
        mask = ~df["nhood"].astype(str).str.strip().str.lower().isin(["", "nan", "undefined", "none", "null"])
        df = df[mask].copy()
    
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
    Map a value to a sequential gradient using a LOGARITHMIC scale.
    This helps visualize data concentrated at the lower end (e.g., 1-10) by spreading the color spectrum.
    """
    if max_val == min_val:
        return hex_to_rgba("#4A90D9", 180)  # Default blue
    
    # 1. Logarithmic Normalization
    # We add 1 to avoid log(0). 
    # Formula: (log(value + 1) - log(min + 1)) / (log(max + 1) - log(min + 1))
    
    # Ensure values are non-negative for log
    val_clamped = max(0, value)
    min_clamped = max(0, min_val)
    max_clamped = max(0, max_val)
    
    log_val = np.log1p(val_clamped)
    log_min = np.log1p(min_clamped)
    log_max = np.log1p(max_clamped)
    
    if log_max == log_min:
        normalized = 0.0
    else:
        normalized = (log_val - log_min) / (log_max - log_min)
        
    normalized = max(0, min(1, normalized))
    
    # Gradient stops (Seattle palette optimized for contrast on dark map)
    # Low (Blue) -> Med (Cyan/Green) -> High (Gold/Red)
    stops = [
        (0.0, "#0046AD"),    # Seattle Blue (Low)
        (0.25, "#63B1E5"),   # Cyan
        (0.50, "#A3D559"),   # Green
        (0.75, "#FECB00"),   # Gold
        (1.0, "#E63946"),    # Red (High Alert)
    ]
    
    for i in range(len(stops) - 1):
        if stops[i][0] <= normalized <= stops[i + 1][0]:
            t = (normalized - stops[i][0]) / (stops[i + 1][0] - stops[i][0])
            # Interpolate
            c1 = stops[i][1]
            c2 = stops[i + 1][1]
            r = int(int(c1[1:3], 16) * (1 - t) + int(c2[1:3], 16) * t)
            g = int(int(c1[3:5], 16) * (1 - t) + int(c2[3:5], 16) * t)
            b = int(int(c1[5:7], 16) * (1 - t) + int(c2[5:7], 16) * t)
            return [r, g, b, 200] # Slightly more opaque
    
    return hex_to_rgba(stops[-1][1], 200)


def create_line_layer_df(df: pd.DataFrame, value_col: str, extra_cols: list = None) -> pd.DataFrame:
    """
    Create DataFrame for PyDeck with path and color for each edge.
    extra_cols: List of additional columns to include in the output DataFrame for tooltips.
    """
    cols_to_keep = ["u_lat", "u_lon", "v_lat", "v_lon", value_col]
    if extra_cols:
        # Only add if they exist in df
        valid_extras = [c for c in extra_cols if c in df.columns]
        cols_to_keep.extend(valid_extras)
    
    valid = df[cols_to_keep].dropna(subset=[value_col, "u_lat", "u_lon", "v_lat", "v_lon"])
    
    if valid.empty:
        return pd.DataFrame()
    
    min_val = valid[value_col].min()
    max_val = valid[value_col].max()
    
    result = []
    for _, row in valid.iterrows():
        color = tableau_gradient(row[value_col], min_val, max_val)
        entry = {
            "path": [[row["u_lon"], row["u_lat"]], [row["v_lon"], row["v_lat"]]],
            "color": color,
            "value": row[value_col]
        }
        # Add extra columns to the entry
        if extra_cols:
            for c in extra_cols:
                if c in row:
                    val = row[c]
                    # Format floats nicely
                    if isinstance(val, float):
                        val = round(val, 2)
                    entry[c] = val
                else:
                    entry[c] = "N/A"
        result.append(entry)
    
    return pd.DataFrame(result)


def render_map(df_map: pd.DataFrame, value_col: str, title: str, show_legend: bool = True, tooltip_html: str = None):
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
    
    # Default tooltip if none provided
    if tooltip_html is None:
        tooltip_html = "<b>Value:</b> {value}"
        
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip={
            "html": tooltip_html,
            "style": {"backgroundColor": "#0046AD", "color": "white", "padding": "8px", "fontFamily": "Open Sans, sans-serif"}
        }
    )
    
    st.pydeck_chart(r, use_container_width=True)
    
    if show_legend:
        min_val = df_map["value"].min()
        max_val = df_map["value"].max()
        # Seattle.gov style gradient legend (Seattle Blue → Light Blue → Green → Gold)
        st.markdown(f"""
        <div style="margin-top: 12px; padding: 12px; background: #FFFFFF; border: 1px solid #63B1E5; border-radius: 8px; font-family: Open Sans, sans-serif;">
            <div style="font-weight: 600; margin-bottom: 6px; color: #000000;">Scale</div>
            <div style="height: 12px; background: linear-gradient(to right, #0046AD, #63B1E5, #A3D559, #FECB00, #E63946); border-radius: 4px; margin-bottom: 6px;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; color: #000000;">
                <span>{min_val:.0f}</span>
                <span>{max_val:.0f}</span>
            </div>
            <div style="font-size: 10px; color: #666; text-align: center; margin-top: 4px;">Log-scaled for better visibility of lower values</div>
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
        df_filtered = df_filtered.sample(n=max_edges, random_state=42)
    
    df_edges = create_line_layer_df(df_filtered, "severitynum_barriers", extra_cols=["nhood"])
    
    # Map 1 Tooltip: Neighbourhood, Barrier Count
    tooltip1 = """
    <b>Neighbourhood:</b> {nhood}<br>
    <b>Barrier Count:</b> {value}
    """
    render_map(df_edges, "severitynum_barriers", "Barrier Count", show_legend=True, tooltip_html=tooltip1)
    
    # Summary stats
    st.subheader("Summary Statistics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Walkable Pathways", f"{len(df):,}")
    with c2:
        edges_with_issues = (df["severitynum_barriers"] > 0).sum()
        st.metric("Pathways with Barriers", f"{edges_with_issues:,}")
    with c3:
        st.metric("Max Barriers (in a single pathway)", f"{df['severitynum_barriers'].max():.0f}")
    with c4:
        st.metric("Avg Barriers (pathways with issues)", f"{df[df['severitynum_barriers']>0]['severitynum_barriers'].mean():.2f}")
    
    # Repair Prioritization for Neighbourhoods
    st.subheader("Repair Prioritization for Neighbourhoods")
    st.markdown("Top 3 neighbourhoods to focus on based on barrier volume and severity.")
    
    if "nhood" in df.columns:
        # Filter and aggregate
        # Exclude NaN or weird neighbourhood names if any
        df_prio = df.dropna(subset=["nhood"]).copy()
        df_prio["nhood"] = df_prio["nhood"].astype(str).str.strip()
        df_prio = df_prio[~df_prio["nhood"].isin(["", "nan", "undefined", "None"])]
        
        prio_stats = df_prio.groupby("nhood").agg(
            total_barriers=("severitynum_barriers", "sum"),
            avg_severity=("severityavg_severity", "mean")
        ).reset_index()
        
        # Priority Score Formula (Raw): (Sum of barriers * 0.5) + (Average Severity * 2)
        prio_stats["raw_score"] = (prio_stats["total_barriers"] * 0.5) + (prio_stats["avg_severity"] * 2)
        
        # Normalize to 0-100 scale
        min_score = prio_stats["raw_score"].min()
        max_score = prio_stats["raw_score"].max()
        
        if max_score > min_score:
            prio_stats["priority_score"] = ((prio_stats["raw_score"] - min_score) / (max_score - min_score)) * 100
        else:
            prio_stats["priority_score"] = 0 if max_score == 0 else 100

        # Rank and get top 3
        top_3 = prio_stats.sort_values("priority_score", ascending=False).head(3)
        
        # Display as cards
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for i, (idx, row) in enumerate(top_3.iterrows()):
            with cols[i]:
                # Custom styled card for prioritization
                st.markdown(f"""
                <div style="
                    padding: 16px;
                    border-radius: 8px;
                    background-color: #F0F4F8;
                    border-left: 5px solid #0046AD;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <h3 style="margin: 0; color: #0046AD; font-size: 1.2rem;">#{i+1} {row['nhood']}</h3>
                    <div style="font-size: 0.9rem; color: #555; margin-top: 8px;">
                        <strong>Priority Score:</strong> {row['priority_score']:.1f}<br>
                        <strong>Total Barriers:</strong> {row['total_barriers']:.0f}<br>
                        <strong>Avg Severity:</strong> {row['avg_severity']:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Neighbourhood data not available for prioritization.")


def issue_metrics_tab(df: pd.DataFrame):
    """Issue Metrics tab - Select label and see map color-coded by label_avg (excluding zeros)."""
    st.header("Issue Metrics by Label")
    st.markdown("""
    Select an issue type below to see where those specific problems exist across Seattle. 
    The map is color-coded by the **average severity** (label_avg) for that issue type.
    **Only edges with non-zero values** for the selected label are shown.
    """)
    
    options_list = list(LABEL_OPTIONS.keys())
    # Try to set 'NoSidewalk' as default index
    default_index = 0
    if "NoSidewalk" in options_list:
        default_index = options_list.index("NoSidewalk")
        
    selected_label = st.selectbox(
        "Select Issue Type",
        options=options_list,
        index=default_index,
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
        df_filtered = df_filtered.sample(n=max_edges, random_state=42)
    
    df_edges = create_line_layer_df(df_filtered, avg_col, extra_cols=["nhood", "severitynum_barriers"])
    
    # Map 2 Tooltip: Neighbourhood, Avg Severity, Barrier Count
    # note: 'value' here is label_avg (avg severity)
    tooltip2 = f"""
    <b>Neighbourhood:</b> {{nhood}}<br>
    <b>Avg Severity of Pathway:</b> {{value}}<br>
    <b>Barrier Count:</b> {{severitynum_barriers}}
    """
    render_map(df_edges, avg_col, f"{selected_label} Average Severity", show_legend=True, tooltip_html=tooltip2)
    
    # Stats for this label
    st.subheader(f"{selected_label} Statistics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Pathways with this issue", f"{len(df_filtered):,}")
    with c2:
        st.metric("Average severity (mean)", f"{df_filtered[avg_col].mean():.2f}")

    
    # Neighbourhood visualizations (group by nhood)
    if "nhood" in df.columns:
        count_col = LABEL_COUNT_COLS[selected_label]
        
        # Robust filtering for this specific chart data
        # (Though load_data helps, we do it again to be 100% sure)
        df_clean = df.dropna(subset=["nhood"]).copy()
        
        mask = ~df_clean["nhood"].astype(str).str.strip().str.lower().isin(["undefined", "nan", "none", "null", ""])
        df_clean = df_clean[mask]
        
        nhood_counts = (
            df_clean.groupby("nhood")[count_col]
            .sum()
            .reset_index()
            .rename(columns={count_col: "total_count"})
        )
        nhood_counts = (
            nhood_counts[nhood_counts["total_count"] > 0]
            .sort_values("total_count", ascending=False)
            .head(6)
            .sort_values("total_count", ascending=True)
        )
        
        if not nhood_counts.empty:
            st.subheader("Neighbourhood ranking by issue count")

            fig = px.bar(
                nhood_counts,
                x="total_count",
                y="nhood",
                orientation="h",
                labels={"total_count": f"{selected_label} count", "nhood": "Neighbourhood"},
                color="total_count",
                # Seattle-like continuous scale: Blue -> Cyan -> Gold
                color_continuous_scale=[[0, "#0046AD"], [0.5, "#63B1E5"], [1, "#FECB00"]],
                title=f"Top Neighbourhoods by {selected_label} Count"
            )
            fig.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="Open Sans, sans-serif", color="#000000", size=12),
                title_font=dict(size=14, color="#000000", family="Open Sans, sans-serif"),
                showlegend=False,
                height=max(400, len(nhood_counts) * 20),
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(
                    title=f"Total {selected_label} count",
                    tickfont=dict(color="#000000"), 
                    title_font=dict(color="#000000")
                ),
                yaxis=dict(
                    title="Neighbourhood",
                    categoryorder="total ascending", 
                    tickfont=dict(color="#000000"), 
                    title_font=dict(color="#000000")
                ),
            )
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True, key=f"nhood_ranking_chart_{selected_label}")


def neighbourhood_analysis_tab(df: pd.DataFrame):
    """Neighbourhood Analysis tab - Heat table and stacked bar chart."""
    st.header("Neighbourhood Analysis")
    
    if "nhood" not in df.columns:
        st.warning("Neighbourhood (nhood) column not found in data.")
        return
    
    df_nhood = df.dropna(subset=["nhood"])
    # Filter out weird neighbourhood names (case-insensitive, stripped)
    mask = ~df_nhood["nhood"].astype(str).str.strip().str.lower().isin(["", "nan", "undefined", "none", "null"])
    df_nhood = df_nhood[mask]
    neighbourhoods = sorted(df_nhood["nhood"].unique().tolist())
    
    # --- Neighbourhood selector at top ---
    selected_nhoods = st.multiselect(
        "Select up to 5 neighbourhoods to compare",
        options=neighbourhoods,
        default=neighbourhoods[:5] if len(neighbourhoods) >= 5 else neighbourhoods,
        max_selections=5,
        key="nhood_compare_select",
    )
    
    # Extra safety: filter out "undefined" if it somehow got selected
    if selected_nhoods:
        selected_nhoods = [n for n in selected_nhoods if str(n).strip().lower() not in ["undefined", "nan", "none", "null", ""]]
    
    if not selected_nhoods:
        st.info("Select at least one neighbourhood to view the visualizations.")
        return
    
    df_selected = df_nhood[df_nhood["nhood"].isin(selected_nhoods)]
    
    # --- Heat Table (filtered by selected neighbourhoods) ---
    st.subheader("1. Neighbourhood Table")
    st.markdown("Mean severity and total barriers by neighbourhood.")
    
    heat_data = (
        df_selected.groupby("nhood")
        .agg(
            mean_severity=("severityavg_severity", "mean"),
            total_barriers=("severitynum_barriers", "sum"),
        )
        .round(2)
        .reset_index()
    )
    heat_data = heat_data.rename(columns={"nhood": "Neighbourhood"})
    
    # Apply heat-style background gradient (Seattle palette - blues and greens)
    # Also attempt to style headers via set_table_styles or CSS injection
    styled = (
        heat_data.style
        .background_gradient(subset=["mean_severity"], cmap="Blues")
        .background_gradient(subset=["total_barriers"], cmap="Greens")
        .format({"mean_severity": "{:.2f}", "total_barriers": "{:,.0f}"})
        .set_table_styles([
            {'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'black')]}
        ])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    
    # --- Stacked Bar Chart (filtered by selected neighbourhoods) ---
    st.subheader("2. Issue Counts by Neighbourhood")
    st.markdown("Barrier type distribution across selected neighbourhoods.")
    
    # prepare data
    chart_data = df_selected.groupby("nhood")[ISSUE_COUNT_COLUMNS].sum().reset_index()
    
    # Melt
    chart_melt = chart_data.melt(
        id_vars="nhood",
        value_vars=ISSUE_COUNT_COLUMNS,
        var_name="issue_type",
        value_name="count"
    )
    
    # Clean up issue type names (remove _count suffix)
    chart_melt["issue_type"] = chart_melt["issue_type"].str.replace("_count", "")
    
    # Filter out any rows with 0 count to keep chart clean (optional but good)
    chart_melt = chart_melt[chart_melt["count"] > 0]
    
    # Robust filtering of neighbourhood names (although load_data does it globaly now)
    valid_nhoods_mask = ~chart_melt["nhood"].astype(str).str.strip().str.lower().isin(["undefined", "nan", "none", "null", ""])
    chart_melt = chart_melt[valid_nhoods_mask]
    
    if chart_melt.empty:
        st.info("No issue data found for the selected neighbourhoods.")
    else:
        # Seattle.gov color palette
        seattle_colors = ["#0046AD", "#63B1E5", "#A3D559", "#FECB00", "#003580", "#4A90A4", "#8B7355"]
        
        fig = px.bar(
            chart_melt,
            x="nhood",
            y="count",
            color="issue_type",
            title="Issue Counts by Neighbourhood", # Set explicit title
            color_discrete_sequence=seattle_colors,
            labels={
                "nhood": "Neighbourhood", 
                "count": "Total Issues", 
                "issue_type": "Issue Type"
            }
        )
        
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Open Sans, sans-serif", color="#000000", size=12),
            title_font=dict(size=16, color="#000000", family="Open Sans, sans-serif"),
            legend_title_text="Issue Type",
            xaxis=dict(
                title="Neighbourhood",
                tickfont=dict(color="#000000"),
                title_font=dict(color="#000000"),
                tickangle=-45
            ),
            yaxis=dict(
                title="Count",
                tickfont=dict(color="#000000"),
                title_font=dict(color="#000000")
            ),
            legend=dict(
                font=dict(color="#000000"),
                bgcolor="rgba(255,255,255,0.5)"
            ),
            margin=dict(t=50, l=50, r=20, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="nhood_stacked_bar_chart_v2")


def main():
    # Seattle.gov / Seattle Public Utilities theme - fonts, colors, styling
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
    /* Seattle.gov typography - Open Sans, white background, black text */
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: "Open Sans", "Segoe UI", Tahoma, sans-serif !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    /* Reduce top padding (remove "shape"/whitespace) */
    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 1rem !important;
        margin-top: 0 !important;
    }
    /* Main title - Seattle Edge Accessibility Dashboard */
    h1, [data-testid="stMarkdown"] h1 {
        color: #000000 !important;
        padding-top: 0.5rem !important; /* Slight padding so it doesn't touch the browser edge */
        margin-top: 0 !important;
    }
    h1, h2, h3 {
        font-family: "Open Sans", "Segoe UI", Tahoma, sans-serif !important;
        color: #000000 !important;
        font-weight: 700 !important;
    }
    .stMarkdown p, .stMarkdown span, .stMarkdown strong {
        font-family: "Open Sans", "Segoe UI", Tahoma, sans-serif !important;
        color: #000000 !important;
    }
    /* Tabs - Seattle Blue active state */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #63B1E5;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: "Open Sans", "Segoe UI", Tahoma, sans-serif !important;
        font-weight: 600 !important;
        padding: 12px 24px;
        color: #000000 !important;
    }
    .stTabs [aria-selected="true"] {
        border-bottom-color: #0046AD !important;
        color: #000000 !important;
    }
    /* Buttons, selectboxes - Seattle Blue */
    .stButton > button {
        font-family: "Open Sans", "Segoe UI", Tahoma, sans-serif !important;
        background-color: #0046AD !important;
        color: white !important;
    }
    .stButton > button:hover {
        background-color: #003580 !important;
        border-color: #003580 !important;
    }
    /* Metrics and data frames - labels and values */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stMetric"] label, [data-testid="stMetric"] p {
        font-family: "Open Sans", "Segoe UI", Tahoma, sans-serif !important;
        color: #000000 !important;
    }
    /* Input widgets */
    .stSelectbox label, .stMultiSelect label {
        font-family: "Open Sans", "Segoe UI", Tahoma, sans-serif !important;
        color: #000000 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #0046AD !important;
        color: white !important;
        border-color: #0046AD !important;
    }
    .stSelectbox div[data-baseweb="select"] > div:hover {
        background-color: #003580 !important;
    }
    .stSelectbox div[data-baseweb="popover"] {
        background-color: #FFFFFF !important;
        color: black !important;
    }
    /* Sidebar if used */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
    }
    /* Info, caption, and other text blocks */
    .stAlert, .stCaption, [data-testid="stCaptionContainer"] {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Seattle Accessibility Dashboard")
    st.markdown(
        "**Make Seattle Accessible.** "
        "Government dashboard for monitoring sidewalk and accessibility issues across Seattle."
    )
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs([
        "**🏠 Home**",
        "**📊 Issue Metrics**",
        "**📊 Neighbourhood Analysis**",
    ])
    
    with tab1:
        home_tab(df)
    
    with tab2:
        issue_metrics_tab(df)
    
    with tab3:
        neighbourhood_analysis_tab(df)


if __name__ == "__main__":
    main()
