import streamlit as st
import pandas as pd
import osmnx as ox
from streamlit_folium import st_folium
import folium

st.set_page_config(layout="wide")

# --- Sidebar for Navigation ---
st.sidebar.title("Seattle Accessibility App")
user_type = st.sidebar.radio("Select User Mode:", ["Pedestrian / Wheelchair User", "Government Official"])

# ==========================================
# MODE 1: USER (Route Finder)
# ==========================================
if user_type == "Pedestrian / Wheelchair User":
    st.title("♿ Safe Route Finder")
    
    col1, col2 = st.columns(2)
    with col1:
        start_loc = st.text_input("From (Start Location)", "Space Needle, Seattle")
    with col2:
        end_loc = st.text_input("To (Destination)", "Pike Place Market, Seattle")

    
    # Initialize session state for route
    if "route_found" not in st.session_state:
        st.session_state.route_found = False

    if st.button("Find Safe Route"):
        st.session_state.route_found = True
        
    if st.session_state.route_found:
        st.write(f"Calculating path from **{start_loc}** to **{end_loc}**...")
        
        # --- PLACEHOLDER FOR YOUR ROUTING LOGIC ---
        # 1. Geocode addresses to Lat/Lon
        # 2. Run Dijkstra's algorithm on your graph (using friction costs)
        # 3. Get the ML predictions for the path
        
        # Helper function to display metrics
        def display_metrics():
            st.subheader("Trip Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Travel Time", "18 mins", "+4 mins delay")
            m2.metric("Difficulty Score", "High (7/10)", "Steep Hills")
            m3.metric("Hazard Probability", "Low (12%)", "Safe Path")
            m4.metric("Slowdowns", "2", "Construction")

        # Helper function to create the map
        def create_map():
            m = folium.Map(location=[47.6062, -122.3321], zoom_start=13)
            folium.Marker([47.6205, -122.3493], popup="Space Needle", tooltip="Start: Space Needle").add_to(m)
            folium.Marker([47.6097, -122.3421], popup="Pike Place Market", tooltip="End: Pike Place Market").add_to(m)
            folium.PolyLine(locations=[[47.6205, -122.3493], [47.6097, -122.3421]], color="blue", weight=5, opacity=0.7).add_to(m)
            return m

        # Create two columns for side-by-side maps
        map_col1, map_col2 = st.columns(2)
        
        with map_col1:
            st.write("**Map View 1**")
            map1 = create_map()
            st_folium(map1, width=700, height=500, key="map1")
            display_metrics()

        with map_col2:
            st.write("**Map View 2**")
            map2 = create_map()
            st_folium(map2, width=700, height=500, key="map2")
            display_metrics()

# ==========================================
# MODE 2: GOVERNMENT (Dashboard)
# ==========================================
elif user_type == "Government Official":
    st.title("🏛️ Infrastructure Dashboard")
    st.markdown("Prioritize repairs based on severity and neighborhood data.")

    # Load your data (cached so it doesn't reload every click)
    @st.cache_data
    def load_data():
        return pd.read_csv('seattle_dataset.csv')
    
    df = load_data()

    # Insight 1: Worst Neighborhoods
    st.subheader("Top 5 Neighborhoods Needs Attention")
    neighborhood_counts = df['neighborhood'].value_counts().head(5)
    st.bar_chart(neighborhood_counts)

    # Insight 2: Severity Breakdown
    st.subheader("Barrier Severity Distribution")
    severity_counts = df['severity'].value_counts()
    st.bar_chart(severity_counts)