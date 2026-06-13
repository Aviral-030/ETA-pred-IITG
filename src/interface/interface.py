import streamlit as st
import pandas as pd
from helper import RoutingEngine 

st.set_page_config(page_title="Delhivery AI Router", page_icon="🚚", layout="wide")

@st.cache_resource
def load_engine():
    return RoutingEngine()

with st.spinner("Initializing AI Logistics Engine..."):
    engine = load_engine()

hub_names = engine.hub_info['hub_name'].tolist()
id_to_name = engine.hub_info.set_index('hub_number')['hub_name'].to_dict()
name_to_row = engine.hub_info.set_index('hub_name')

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4305/4305315.png", width=80)
    st.title("Routing Parameters")
    st.markdown("---")
    
    ui_vehicle_radio = st.radio(
        "🚛 Vehicle Strategy", 
        options=["Auto (Fastest)", "FTL", "Carting"],
        help="Auto allows the AI to mix truck types per leg to find the absolute fastest route."
    )
    
    st.markdown("---")
    leave_now = st.checkbox("🕒 Leave Now (Live Server Time)", value=True)
    if leave_now:
        ui_time_slider = -1
    else:
        ui_time_slider = st.slider("Select Dispatch Hour", min_value=0.0, max_value=23.99, value=12.0, step=0.5)
        
    st.markdown("---")
    ui_multiselect_blacklist = st.multiselect(
        "🚫 Blacklist Hubs (Avoid)", 
        options=hub_names,
        help="These nodes will be physically removed from the routing graph."
    )

st.title("🚚 AI Intelligent Routing Dashboard")
st.markdown("Select your dispatch and arrival hubs to evaluate optimal supply chain pathways.")

def display_hub_stats(selected_hub_name, card_title):
    hub_data = name_to_row.loc[selected_hub_name]
    
    st.markdown(f"### {card_title}")
    
    if hub_data['bottleneck_score'] > 0.05:
        st.warning(f"⚠️ **High Traffic Warning:** {selected_hub_name} has a bottleneck score of {hub_data['bottleneck_score']:.3f}. Expect potential local delays.")
    elif hub_data['delay_ratio'] > 1.5:
         st.warning(f"⏳ **Historical Delay Warning:** This hub typically operates at {hub_data['delay_ratio']:.2f}x standard processing time.")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Bottleneck Score", f"{hub_data['bottleneck_score']:.3f}")
    c2.metric("Betweenness Centrality", f"{hub_data['betweenness_centrality']:.4f}")
    c3.metric("Avg Delay Ratio", f"{hub_data['delay_ratio']:.2f}x")

col_source, col_dest = st.columns(2)

with col_source:
    with st.container(border=True):
        ui_source_dropdown = st.selectbox("🟢 Departure Hub", options=hub_names, index=0)
        st.divider()
        display_hub_stats(ui_source_dropdown, "Source Analytics")

with col_dest:
    with st.container(border=True):
        ui_dest_dropdown = st.selectbox("🔴 Destination Hub", options=hub_names, index=1 if len(hub_names) > 1 else 0)
        st.divider()
        display_hub_stats(ui_dest_dropdown, "Destination Analytics")


st.markdown("<br>", unsafe_allow_html=True) # Spacer

if st.button("🚀 Calculate Optimal Route", type="primary", use_container_width=True):
    
    if ui_source_dropdown == ui_dest_dropdown:
        st.error("Source and Destination cannot be the same hub. Please select different hubs.")
    else:
        user_payload = {
            "source_name": ui_source_dropdown,
            "destination_name": ui_dest_dropdown,
            "is_ftl": 1 if ui_vehicle_radio == "FTL" else (0 if ui_vehicle_radio == "Carting" else -1),
            "is_carting": 1 if ui_vehicle_radio == "Carting" else (0 if ui_vehicle_radio == "FTL" else -1),
            "start_hour": ui_time_slider, 
            "blacklist_hubs": ui_multiselect_blacklist
        }
        
        with st.spinner("AI is evaluating millions of supply chain parameters..."):
            result = engine.process_route_request(user_payload)
            
            if "error" in result:
                st.error(result["error"])
            elif result["best_path"] is None:
                st.error("Route calculation failed. AI model rejected all physical paths due to constraints.")
            else:
                st.markdown("---")
                st.header("🏆 AI Recommended Route")
                
                best_path_names = [id_to_name[node_id] for node_id in result['best_path']]
                
                st.metric("Total Predicted Transit Time", f"{result['best_eta_hours']} Hours")
                
                formatted_path = " ➔ \n\n".join([f"`{name}`" for name in best_path_names])
                st.info(formatted_path)

                
                all_paths = sorted(result['all_evaluated_paths'], key=lambda x: x['total_eta'])
                
                alternatives = all_paths[1:4] 
                
                if alternatives:
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander("Explore Alternative Routes"):
                        for i, alt in enumerate(alternatives):
                            alt_names = [id_to_name[node_id] for node_id in alt['path']]
                            alt_formatted = " ➔ ".join(alt_names)
                            st.write(f"**Alternative {i+1}** | ⏱️ ETA: **{alt['total_eta']:.2f} Hours**")
                            st.caption(alt_formatted)
                            st.divider()