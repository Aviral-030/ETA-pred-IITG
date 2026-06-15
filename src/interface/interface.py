import streamlit as st
import pandas as pd
from helper import RoutingEngine

st.set_page_config(
    page_title="Delhivery Routing Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_engine():
    return RoutingEngine()

engine = load_engine()

# Map lookups
hub_names = engine.hub_info["hub_name"].tolist()
id_to_name = engine.hub_info.set_index("hub_number")["hub_name"].to_dict()
name_to_row = engine.hub_info.set_index("hub_name")

# Sidebar Controls
with st.sidebar:
    st.title("Route Controls")

    source = st.selectbox("Departure Hub", hub_names)
    destination = st.selectbox(
        "Destination Hub",
        hub_names,
        index=1 if len(hub_names) > 1 else 0
    )

    vehicle = st.selectbox("Vehicle Strategy", ["Auto (Fastest)", "FTL", "Carting"])
    
    leave_now = st.checkbox("Leave Now", value=True)
    dispatch_time = -1 if leave_now else st.slider("Dispatch Hour", 0.0, 23.5, 12.0, 0.5)

    blacklist = st.multiselect("Avoid Hubs", hub_names)

    calculate = st.button("Optimize Route", use_container_width=True, type="primary")

# Main View
st.title("Delhivery Routing Intelligence")

def risk_label(score):
    if score > 0.50: return "High"
    if score > 0.15: return "Medium"
    return "Low"

st.subheader("Hub Analytics")
col1, col2 = st.columns(2)

for col, hub, title in [(col1, source, "Source"), (col2, destination, "Destination")]:
    data = name_to_row.loc[hub]
    with col:
        # st.container(border=True) acts as a lifted card
        with st.container(border=True):
            st.markdown(f"**{title}: {hub}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Bottleneck", f"{data['bottleneck_score']:.3f}")
            c2.metric("Delay Ratio", f"{data['delay_ratio']:.2f}x")
            c3.metric("Centrality", f"{data['betweenness_centrality']:.4f}")

st.divider()

# Execution Logic
if calculate:
    if source == destination:
        st.error("Source and destination cannot be identical.")
        st.stop()

    payload = {
        "source_name": source,
        "destination_name": destination,
        "is_ftl": 1 if vehicle == "FTL" else (0 if vehicle == "Carting" else -1),
        "is_carting": 1 if vehicle == "Carting" else (0 if vehicle == "FTL" else -1),
        "start_hour": dispatch_time,
        "blacklist_hubs": blacklist,
    }

    with st.spinner("Evaluating network paths..."):
        result = engine.process_route_request(payload)

    if "error" in result:
        st.error(result["error"])
        st.stop()

    # Process Results
    best_path_names = [id_to_name[x] for x in result["best_path"]]
    risk_score = name_to_row.loc[best_path_names]["bottleneck_score"].mean()

    # Recommended Route Summary - Lifted Card
    with st.container(border=True):
        st.subheader("Recommended Route Summary")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ETA", f"{result['best_eta_hours']} hrs")
        m2.metric("Hops", len(best_path_names) - 1)
        m3.metric("Risk Level", risk_label(risk_score))
        m4.metric("Routes Evaluated", len(result["all_evaluated_paths"]))

        st.markdown("**Route Timeline:**")
        st.code(" ➔ ".join(best_path_names), language="text")

    st.markdown("<br>", unsafe_allow_html=True) # Tiny bit of spacing

    # Alternative Routes
    st.subheader("Alternative Route Comparison")
    
    all_paths = sorted(result["all_evaluated_paths"], key=lambda x: x["total_eta"])
    rows = [
        {
            "Rank": rank,
            "ETA (Hours)": round(path["total_eta"], 2),
            "Route": " → ".join(id_to_name[x] for x in path["path"])
        }
        for rank, path in enumerate(all_paths, start=1)
    ]

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)