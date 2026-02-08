# Accessible Seattle – Accessibility-Aware Urban Routing and Analytics

## Project Overview
**Accessible Seattle** is an accessibility-first urban navigation and analytics system designed to improve mobility for wheelchair users and individuals with physical or visual impairments. While traditional navigation systems optimize for speed and distance, this platform prioritizes infrastructure quality and safety.

By modeling the city's sidewalk network as a weighted graph, the system identifies obstacles, evaluates surface conditions, and accounts for missing curb ramps to compute routes that minimize physical risk.

---

## Key Features

### 1. User-Facing Routing Mode
- **Risk-Minimized Navigation:** Computes paths based on a cumulative accessibility risk score rather than just shortest distance.
- **Barrier Awareness:** Explicitly avoids segments with steep slopes, poor surface quality, or missing curb ramps.
- **Interactive Visualization:** Displays routes on a map highlighting safe and unsafe segments with real-time route-level metrics (e.g., average risk, high-risk segment count).

### 2. Government & Planning Mode
- **Geospatial Analytics:** Aggregates sidewalk data to visualize spatial patterns of accessibility deficits across the city.
- **Data-Driven Decisions:** Helps urban planners identify high-risk areas and prioritize infrastructure investments (e.g., where to install new curb ramps).
- **High-Scale Visualization:** Uses 3D geospatial mapping to display city-wide risk distributions.

---

## Technical Implementation

- **Graph Modeling:** Built using **NetworkX**, treating intersections as nodes and sidewalk segments as edges enriched with accessibility metadata.
- **Frontend Interface:** Developed with **Streamlit** for a responsive, interactive web experience.
- **Geospatial Visualization:** 
  - **Folium** for detailed, route-level 2D maps.
- **Data Processing:** **Pandas** and **NumPy** for efficient handling of municipal sidewalk datasets and risk score calculations.
- **Machine Learning (Experimental):** Exploration of **Graph Neural Networks (GNNs)** to predict accessibility risks on edges with missing or incomplete data.

---

## Methodology: The Risk Score
Each sidewalk segment is assigned a score based on:
- **Surface Quality:** Smoothness and condition of the pavement.
- **Curb Ramps:** Presence and compliance of ramps at intersections.
- **Obstacles:** Fixed barriers like poles, narrow passages, or construction.
- **Slope:** Incline percentage, which significantly impacts manual wheelchair users.

---
## Project Structure
Datahon-shared-repo-main/
│
├── main.py                     # Streamlit application entry point
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── seattle_dataset.csv
│   ├── blind_assitance.csv
│   ├── mobility_issue_assistance.csv
│   ├── normal_user.csv
│   └── my_plot.png
│
├── export/
│   └── gnn_input.csv           # Preprocessed graph edge data
│
├── notebooks/
│   ├── GNN.ipynb
│   ├── graph_data.ipynb
│   ├── seattle_data_prep.ipynb
│   └── speech_recognition.ipynb
│
└── .gitignore
---

## Future Roadmap
- **Real-Time Reporting:** Crowdsourced reporting for temporary obstacles like construction or debris.
- **Voice-First Interaction:** Optimized navigation cues for visually impaired users.
- **City Expansion:** Scaling the graph model to other major metropolitan areas beyond Seattle.
- **Workflow Integration:** Direct integration with municipal maintenance ticketing systems.

---

## Impact
This project shifts the focus of urban navigation from **efficiency** to **inclusivity**, demonstrating how data engineering and graph algorithms can be leveraged to address real-world mobility challenges and create more equitable cities.