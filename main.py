import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import osmnx as ox
from streamlit_folium import st_folium
import folium
import requests
from streamlit_searchbox import st_searchbox
import networkx as nx
from pathlib import Path

st.set_page_config(layout="wide")
# Add this near the top of your file, after st.set_page_config()

def add_seattle_rain_animation():
    """Add Seattle-themed rain animation to the background"""
    st.markdown("""
    <style>
    /* Rain animation */
    .rain-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    
    .raindrop {
        position: absolute;
        width: 2px;
        height: 50px;
        background: linear-gradient(to bottom, rgba(174, 194, 224, 0.5), transparent);
        animation: fall linear infinite;
    }
    
    @keyframes fall {
        to {
            transform: translateY(100vh);
        }
    }
    
    /* Make main content appear above rain */
    .main > div {
        position: relative;
        z-index: 1;
    }
    
    /* Optional: Add subtle Seattle fog effect */
    body {
        background: linear-gradient(to bottom, #2c3e50 0%, #34495e 100%);
    }
    </style>
    
    <div class="rain-container" id="rain"></div>
    
    <script>
    // Generate raindrops
    const rainContainer = document.getElementById('rain');
    if (rainContainer && rainContainer.children.length === 0) {
        for (let i = 0; i < 50; i++) {
            const drop = document.createElement('div');
            drop.className = 'raindrop';
            drop.style.left = Math.random() * 100 + '%';
            drop.style.animationDuration = (Math.random() * 1 + 0.5) + 's';
            drop.style.animationDelay = Math.random() * 2 + 's';
            rainContainer.appendChild(drop);
        }
    }
    </script>
    """, unsafe_allow_html=True)

# Call this function right after st.set_page_config()
add_seattle_rain_animation()
# --- Sidebar for Navigation ---
# --- Sidebar for Navigation ---
# Custom CSS to inject for styling the buttons if needed (optional)
st.markdown("""
<style>
div.stButton > button:first-child {
    height: 100px;
    width: 100%;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "user_mode" not in st.session_state:
    st.session_state.user_mode = None

# --- LANDING PAGE ---
if st.session_state.user_mode is None:
    st.title("Welcome to Seattle Accessibility App")
    st.subheader("Select your mode to continue:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚶 Pedestrian / Wheelchair User"):
            st.session_state.user_mode = "Pedestrian"
            st.rerun()
            
    with col2:
        if st.button("🏛️ Government Official"):
            st.session_state.user_mode = "Government"
            st.rerun()

# --- MAIN APP LOGIC ---
else:
    # Sidebar back button
    if st.sidebar.button("← Back to Home"):
        st.session_state.user_mode = None
        st.rerun()

    user_type = st.session_state.user_mode

# ==========================================
# MODE 1: USER (Route Finder)
# ==========================================
if st.session_state.user_mode == "Pedestrian":
    st.title("♿ Safe Route Finder")
    
    # Load graph with GNN predictions (cached)
    @st.cache_resource
    def load_graph_with_predictions(dataset_file, pred_column):
        """Load Seattle street network and attach GNN risk predictions."""
        import pickle
        import os
        
        # Use dataset-specific cache file
        graph_cache_file = f"seattle_graph_{dataset_file.replace('.csv', '')}.pkl"
        
        # Check if cached graph exists
        if os.path.exists(graph_cache_file):
            with st.spinner("Loading cached graph... (~2 seconds)"):
                try:
                    with open(graph_cache_file, "rb") as f:
                        G_full = pickle.load(f)
                    st.success("✅ Loaded from cache!")
                    return G_full
                except Exception as e:
                    st.warning(f"Cache file corrupted, rebuilding... ({e})")
        
        # Build graph from scratch if no cache
        with st.spinner("🌐 Downloading Seattle street network from OpenStreetMap... (~15 seconds, only happens once)"):
            # Download Seattle walk network
            G_full = ox.graph_from_place("Seattle, Washington, USA", network_type="walk")
            G_full = nx.Graph(G_full)
        
        with st.spinner(f"📊 Loading GNN predictions from {dataset_file} and attaching to graph... (~5 seconds)"):
            # Load GNN predictions from selected dataset
            df = pd.read_csv(dataset_file)
            
            # Create coordinate key function
            def key(lat, lon):
                return (round(lat, 6), round(lon, 6))
            
            # Map: (u_coord, v_coord) -> row with predictions
            edge_features = {}
            for _, r in df.iterrows():
                u = key(r.u_lat, r.u_lon)
                v = key(r.v_lat, r.v_lon)
                edge_features[(u, v)] = r
                edge_features[(v, u)] = r  # undirected
            
            # Attach predictions to graph edges
            for u, v in G_full.edges():
                u_coord = key(G_full.nodes[u]["y"], G_full.nodes[u]["x"])
                v_coord = key(G_full.nodes[v]["y"], G_full.nodes[v]["x"])
                
                data_row = edge_features.get((u_coord, v_coord), None)
                
                if data_row is not None:
                    # Use GNN prediction as risk_norm
                    G_full[u][v]["risk_norm"] = float(data_row[pred_column]) * 10.0  # Scale to 0-10
                else:
                    # Default: safe edge
                    G_full[u][v]["risk_norm"] = 0.0
                
                # Calculate wheelchair cost
                length = G_full[u][v].get("length", 1.0)
                risk_norm = G_full[u][v]["risk_norm"]
                alpha = 1.0  # distance weight
                beta = 2.0   # risk weight
                G_full[u][v]["wheelchair_cost"] = alpha * length + beta * risk_norm
        
        # Save to cache for next time
        with st.spinner("💾 Saving graph to cache for faster future loads..."):
            try:
                with open(graph_cache_file, "wb") as f:
                    pickle.dump(G_full, f)
                st.success("✅ Graph cached! Next load will be ~2 seconds.")
            except Exception as e:
                st.warning(f"Could not save cache: {e}")
        
        return G_full
    
    # Function to search LocationIQ API
    def search_locationiq(searchterm: str) -> list[any]:
        if not searchterm:
            return []
        
        # API Key provided by user
        LOCATIONIQ_API_KEY = "pk.64abdedac2e6a88a6c8af166a80d883f"
        
        url = "https://api.locationiq.com/v1/autocomplete"
        params = {
            "key": LOCATIONIQ_API_KEY,
            "q": searchterm,
            "limit": 5,
            "countrycodes": "us",
            "lat": 47.6062, 
            "lon": -122.3321,
            "dedupe": 1
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            suggestions = []
            for item in data:
                display_name = item.get("display_name")
                lat = item.get("lat")
                lon = item.get("lon")
                if display_name and lat and lon:
                    suggestions.append((display_name, (float(lat), float(lon))))
            
            return suggestions
        except Exception as e:
            return []

    # User type selection (Normal User, Blind Assistance, Mobility Issue)
    st.subheader("Select User Type")
    user_type_option = st.radio(
        "Choose the accessibility profile for route optimization:",
        options=["🚶 Normal User", "👁️ Blind Assistance", "♿ Mobility Issue"],
        index=0,
        horizontal=True,
        key="user_type_selection"
    )
    
    # Map user selection to dataset file, prediction column, and safety percentile threshold
    dataset_mapping = {
        "🚶 Normal User": ("normal_user.csv", "pred_normal", 90),  # Top 5% are dangerous
        "👁️ Blind Assistance": ("blind_assitance.csv", "pred_blind", 80),  # Top 10% are dangerous
        "♿ Mobility Issue": ("mobility_issue_assistance.csv", "pred", 85)  # Top 15% are dangerous
    }
    selected_dataset, pred_column, safety_percentile = dataset_mapping[user_type_option]
    
    st.info(f"ℹ️ Using dataset: **{selected_dataset}** for route calculation")
    st.markdown("---")
    
    # Detect user type change and auto-trigger route recalculation
    if "prev_user_type" not in st.session_state:
        st.session_state.prev_user_type = user_type_option
    
    user_type_changed = (st.session_state.prev_user_type != user_type_option)
    if user_type_changed:
        st.session_state.prev_user_type = user_type_option
        # Set flag to auto-calculate route
        st.session_state.auto_calculate = True

    # Default coordinates for demonstration (users can still change them)
    DEFAULT_START = ("Magnolia Park, 1461, Magnolia Boulevard West, Seattle, WA, 98199", (47.63378855, -122.39837658))
    DEFAULT_DESTINATION = ("All City Fence Co., 36, South Hudson Street, Georgetown, Seattle, King County, Washington, 98134, United States", (47.5578313, -122.33716503))
    
    # Initialize default coordinates in session state if not already set
    if "start_coords" not in st.session_state:
        st.session_state.start_coords = DEFAULT_START
    if "end_coords" not in st.session_state:
        st.session_state.end_coords = DEFAULT_DESTINATION
    
    # Show default values to user
    st.info(f"📍 **Default Route:** From Magnolia Park to All City Fence Co. (Georgetown). You can change these locations below.")
    
    col1, col2 = st.columns(2)
    with col1:
        start_loc_selection = st_searchbox(
            search_locationiq,
            key="start_loc_search",
            label="From (Start Location)",
            placeholder="Magnolia Park, 1461, Magnolia Boulevard West, Seattle, WA, 98199",
            default=None
        )
    with col2:
        end_loc_selection = st_searchbox(
            search_locationiq,
            key="end_loc_search",
            label="To (Destination)",
            placeholder="All City Fence Co., 36, South Hudson Street, Georgetown, Seattle",
            default=None
        )

    # Store coordinates in session state (only update if user selects new location)
    if start_loc_selection:
        st.session_state.start_coords = start_loc_selection
    if end_loc_selection:
        st.session_state.end_coords = end_loc_selection
    
    # Display current selections
    st.markdown("**Current Selection:**")
    col_a, col_b = st.columns(2)
    with col_a:
        if isinstance(st.session_state.start_coords, tuple) and len(st.session_state.start_coords) == 2:
            if isinstance(st.session_state.start_coords[1], tuple):
                st.caption(f"🟢 From: {st.session_state.start_coords[0]}")
            else:
                st.caption(f"🟢 From: {st.session_state.start_coords}")
    with col_b:
        if isinstance(st.session_state.end_coords, tuple) and len(st.session_state.end_coords) == 2:
            if isinstance(st.session_state.end_coords[1], tuple):
                st.caption(f"🔴 To: {st.session_state.end_coords[0]}")
            else:
                st.caption(f"🔴 To: {st.session_state.end_coords}")
    
    # Only save to JSON when both are selected (avoid writing file on every rerun)
    # This prevents unnecessary file I/O that could cause glitches
    
    # Check if we should auto-calculate (when user type changes)
    should_calculate = st.button("Find Safe Route")
    if "auto_calculate" in st.session_state and st.session_state.auto_calculate:
        should_calculate = True
        st.session_state.auto_calculate = False  # Reset flag after triggering

    if should_calculate:
        if "start_coords" not in st.session_state or "end_coords" not in st.session_state:
            st.error("Please select both start and destination locations.")
        else:
            import math  # Import here for route calculation
            
            # Load graph with selected dataset
            G_full = load_graph_with_predictions(selected_dataset, pred_column)
            
            # Extract coordinates - handle both tuple formats
            # st_searchbox returns (display_name, (lat, lon))
            if isinstance(st.session_state.start_coords, tuple) and len(st.session_state.start_coords) == 2:
                if isinstance(st.session_state.start_coords[1], tuple):
                    # Format: (display_name, (lat, lon))
                    start_display = st.session_state.start_coords[0]
                    start_lat, start_lon = st.session_state.start_coords[1]
                else:
                    # Format: (lat, lon)
                    start_lat, start_lon = st.session_state.start_coords
                    start_display = f"{start_lat}, {start_lon}"
            else:
                st.error("Invalid start coordinate format")
                st.stop()
                
            if isinstance(st.session_state.end_coords, tuple) and len(st.session_state.end_coords) == 2:
                if isinstance(st.session_state.end_coords[1], tuple):
                    # Format: (display_name, (lat, lon))
                    end_display = st.session_state.end_coords[0]
                    end_lat, end_lon = st.session_state.end_coords[1]
                else:
                    # Format: (lat, lon)
                    end_lat, end_lon = st.session_state.end_coords
                    end_display = f"{end_lat}, {end_lon}"
            else:
                st.error("Invalid end coordinate format")
                st.stop()
            
            with st.spinner(f"Finding safest route from **{start_display}** to **{end_display}**..."):
                try:
                    # Find nearest nodes
                    orig_node = ox.distance.nearest_nodes(G_full, X=start_lon, Y=start_lat)
                    dest_node = ox.distance.nearest_nodes(G_full, X=end_lon, Y=end_lat)
                    
                    # Euclidean distance heuristic for A*
                    def euclidean_distance(u, v):
                        y1, x1 = G_full.nodes[u]["y"], G_full.nodes[u]["x"]
                        y2, x2 = G_full.nodes[v]["y"], G_full.nodes[v]["x"]
                        return math.sqrt((y1 - y2)**2 + (x1 - x2)**2)
                    
                    # A* using wheelchair_cost (AVOIDS high-risk edges)
                    path_nodes = nx.astar_path(
                        G_full,
                        source=orig_node,
                        target=dest_node,
                        heuristic=lambda u, v: euclidean_distance(u, v),
                        weight="wheelchair_cost"
                    )
                    
                    # Calculate dynamic risk threshold based on safety_percentile
                    all_risks = [G_full[u][v].get("risk_norm", 0.0) for u, v in G_full.edges()]
                    risk_threshold = np.percentile(all_risks, safety_percentile)
                    
                    # Extract route metrics
                    route_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
                    route_risks = []
                    route_lengths = []
                    total_length = 0
                    high_risk_count = 0
                    
                    for u, v in route_edges:
                        edge_data = G_full[u][v]
                        risk_norm = edge_data.get("risk_norm", 0.0)
                        length = edge_data.get("length", 0.0)
                        
                        route_risks.append(risk_norm)
                        route_lengths.append(length)
                        total_length += length
                        if risk_norm >= risk_threshold:
                            high_risk_count += 1
                    
                    avg_risk = np.mean(route_risks) if route_risks else 0
                    num_edges = len(route_edges)
                    
                    # Create Folium map
                    m = folium.Map(
                        location=[(start_lat + end_lat) / 2, (start_lon + end_lon) / 2],
                        zoom_start=13,
                        tiles='CartoDB Positron',  # Light grayscale map
                        attr='CartoDB'
                    )
                    
                    # Plot safest route (thick BLUE line)
                    route_coords_map = [[G_full.nodes[n]["y"], G_full.nodes[n]["x"]] for n in path_nodes]
                    folium.PolyLine(
                        locations=route_coords_map,
                        color="blue",
                        weight=8,
                        opacity=0.95,
                        tooltip=f"🛡️ SAFEST ROUTE (Avg Risk: {avg_risk:.1f}/10)"
                    ).add_to(m)
                    
                    # Add all edges (green = safe, red = high risk)
                    sample_edges = list(G_full.edges())[:15000]  # Larger sample for visibility at all zoom levels
                    for u, v in sample_edges:
                        edge_data = G_full[u][v]
                        risk_norm = edge_data.get("risk_norm", 0.0)
                        
                        # Above threshold = red (dangerous), rest = green (safe)
                        if risk_norm >= risk_threshold:
                            color = "red"
                            weight = 2
                            opacity = 0.4
                        else:
                            color = "green"
                            weight = 1
                            opacity = 0.2
                        
                        coords = [
                            [G_full.nodes[u]["y"], G_full.nodes[u]["x"]],
                            [G_full.nodes[v]["y"], G_full.nodes[v]["x"]]
                        ]
                        folium.PolyLine(
                            locations=coords,
                            color=color,
                            weight=weight,
                            opacity=opacity
                        ).add_to(m)
                    
                    # Start/End markers
                    folium.Marker(
                        [start_lat, start_lon], 
                        popup=f"🚩 START: {start_display}",
                        icon=folium.Icon(color="green", icon="play")
                    ).add_to(m)
                    folium.Marker(
                        [end_lat, end_lon], 
                        popup=f"🏁 END: {end_display}",
                        icon=folium.Icon(color="red", icon="stop")
                    ).add_to(m)
                    
                    # Legend with route metrics (improved text visibility)
                    danger_pct = 100 - safety_percentile  # Top X% that are dangerous
                    safe_pct = safety_percentile  # Bottom X% that are safe
                    legend_html = f'''
                    <div style="position: fixed; 
                         bottom: 50px; left: 50px; width: 280px; height: 160px; 
                         background-color: white; border:2px solid #333; z-index:9999; 
                         font-size:14px; padding: 15px; color: #000; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
                    <p style="margin: 5px 0; color: #000;"><span style="color: green; font-weight: bold;">🟢 Green:</span> Safe ({safe_pct}% edges)</p>
                    <p style="margin: 5px 0; color: #000;"><span style="color: red; font-weight: bold;">🔴 Red:</span> DANGER (TOP {danger_pct}%)</p>
                    <p style="margin: 5px 0; color: #000;"><span style="color: blue; font-weight: bold;">🔵 Blue:</span> YOUR SAFEST ROUTE</p>
                    <hr style="border-color: #ccc;">
                    <p style="margin: 5px 0; color: #000; font-weight: bold;">📊 ROUTE METRICS:</p>
                    <p style="margin: 5px 0; color: #000;">Avg Risk: <span style="color:#0066cc; font-weight: bold;">{avg_risk:.1f}/10</span></p>
                    <p style="margin: 5px 0; color: #000;">Length: <span style="color:#006600; font-weight: bold;">{total_length:.0f}m</span></p>
                    <p style="margin: 5px 0; color: #000;">High Risk Edges: <span style="color:#ff6600; font-weight: bold;">{high_risk_count}</span></p>
                    </div>
                    '''
                    m.get_root().html.add_child(folium.Element(legend_html))
                    
                    # Store results in session state to persist across reruns
                    st.session_state.route_map = m
                    st.session_state.route_metrics = {
                        "avg_risk": avg_risk,
                        "total_length": total_length,
                        "num_edges": num_edges,
                        "high_risk_count": high_risk_count,
                        "start_display": start_display,
                        "end_display": end_display
                    }
                    st.session_state.route_calculated = True
                    
                except nx.NetworkXNoPath:
                    st.error("❌ No route found between these locations. Please try different points.")
                    st.session_state.route_calculated = False
                except Exception as e:
                    st.error(f"❌ Error calculating route: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.session_state.route_calculated = False
    
    # Display route results from session state (persists across reruns)
    if "route_calculated" in st.session_state and st.session_state.route_calculated:
        metrics = st.session_state.route_metrics
        
        # Display map (returned_objects=[] prevents interaction state from causing reruns)
        st.success(f"✅ Safest route found! ({metrics['num_edges']} segments)")
        st_folium(st.session_state.route_map, width=1400, height=600, key="route_map_display", returned_objects=[])
        
        # Display metrics
        st.subheader("📊 Route Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Risk Score", f"{metrics['avg_risk']:.1f}/10", 
                     "Lower is safer" if metrics['avg_risk'] < 5 else "Use caution")
        with col2:
            st.metric("Total Distance", f"{metrics['total_length']:.0f} m", 
                     f"{metrics['total_length']/1000:.1f} km")
        with col3:
            st.metric("Number of Segments", metrics['num_edges'])
        with col4:
            safety_score = 100 * (1 - metrics['high_risk_count']/max(metrics['num_edges'], 1))
            st.metric("Safety Score", f"{safety_score:.1f}%", 
                     "Excellent!" if safety_score > 95 else "Good")
        
        # Print summary
        st.info(f"""
        **🛤️ Route Complete!**
        - This is the safest wheelchair-accessible route based on AI predictions
        - The route avoids high-risk edges (red) wherever possible
        - Average risk: {metrics['avg_risk']:.2f}/10 (Lower is better)
        - High-risk segments: {metrics['high_risk_count']} out of {metrics['num_edges']} ({100*metrics['high_risk_count']/max(metrics['num_edges'],1):.1f}%)
        """)

# ==========================================
# MODE 2: GOVERNMENT (Dashboard)
# ==========================================
elif st.session_state.user_mode == "Government":
    
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
    
    # Data path
    DATA_PATH = Path("gnn_input.csv")
    
    def load_gnn_data():
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
            map_style="light",
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
    
    
    # Load data
    df = load_gnn_data()
    if df is None:
        st.stop()
    
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