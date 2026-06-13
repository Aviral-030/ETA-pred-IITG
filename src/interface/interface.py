import streamlit as st
from helper import RoutingEngine # Make sure this matches your filename!

# 1. Page Config
st.set_page_config(page_title="AI Logistics Router", layout="wide")
st.title("🚚 AI Logistics Routing Engine")

# 2. Initialize Engine
@st.cache_resource
def load_engine():
    return RoutingEngine()

with st.spinner("Booting AI Engine..."):
    engine = load_engine()

# Extract the list of hub names to populate our dropdowns
hub_names = engine.hub_info['hub_name'].tolist()

# 3. Build the UI Layout
st.subheader("Route Configuration")
col1, col2 = st.columns(2)

with col1:
    ui_source_dropdown = st.selectbox("Source Hub", options=hub_names, index=0)
    
    # Radio buttons for vehicle choice
    ui_vehicle_radio = st.radio(
        "Vehicle Preference", 
        options=["Auto (Fastest)", "FTL", "Carting"],
        help="Auto will simulate both and pick the fastest truck per leg."
    )
    
    # Time logic
    leave_now = st.checkbox("Leave Now (Use Live Server Time)", value=True)
    if leave_now:
        ui_time_slider = -1
    else:
        ui_time_slider = st.slider("Select Start Hour", min_value=0.0, max_value=23.99, value=12.0, step=0.5)

with col2:
    # Default to the second item so source and dest aren't the same on boot
    ui_dest_dropdown = st.selectbox("Destination Hub", options=hub_names, index=1 if len(hub_names) > 1 else 0)
    
    ui_multiselect_blacklist = st.multiselect(
        "Blacklist Hubs", 
        options=hub_names,
        help="The routing algorithm will completely avoid these hubs."
    )

st.markdown("---")

# 4. The Execution Block
if st.button("Calculate Optimal Route", type="primary"):
    
    if ui_source_dropdown == ui_dest_dropdown:
        st.warning("Source and Destination cannot be the same hub.")
    else:
        # Build the exact payload our engine expects
        user_payload = {
            "source_name": ui_source_dropdown,
            "destination_name": ui_dest_dropdown,
            "is_ftl": 1 if ui_vehicle_radio == "FTL" else (0 if ui_vehicle_radio == "Carting" else -1),
            "is_carting": 1 if ui_vehicle_radio == "Carting" else (0 if ui_vehicle_radio == "FTL" else -1),
            "start_hour": ui_time_slider, 
            "blacklist_hubs": ui_multiselect_blacklist
        }
        
        with st.spinner("AI evaluating supply chain network..."):
            result = engine.process_route_request(user_payload)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Fastest AI Predicted ETA: {result['best_eta_hours']} hours")
                
                # Display the path beautifully
                st.write("**Optimal Node Sequence:**")
                st.info(" ➔ ".join([str(node) for node in result['best_path']]))