import networkx as nx
import pandas as pd
import joblib
from datetime import datetime
import itertools
import os
from runner import predict_eta

class RoutingEngine:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../.."))

        graph_path = os.path.join(project_root, 'dataset', 'osrm_graph.pkl')
        hub_info_path = os.path.join(project_root, 'dataset', 'hub-features.csv')
        print("Initializing Routing Engine...")
        self.osrm_graph = joblib.load(graph_path)
        self.hub_info = pd.read_csv(hub_info_path)
        print("Routing Engine Ready.")

    def _get_hub_number(self, hub_name):
        """Safely converts a string name to its integer ID."""
        matching_values = self.hub_info.loc[self.hub_info['hub_name'] == hub_name, 'hub_number'].values
        if len(matching_values) > 0:
            return int(matching_values[0])
        return None

    def _get_candidate_paths(self, src, dest, blacklisted_hubs, k=5):
        """Generates top K paths on a temporary graph that respects blacklists."""
        working_graph = self.osrm_graph.copy()
        
        for hub_name in blacklisted_hubs:
            hub_id = self._get_hub_number(hub_name)
            if hub_id is not None and working_graph.has_node(hub_id):
                working_graph.remove_node(hub_id)

        try:
            paths_generator = nx.shortest_simple_paths(working_graph, src, dest, weight='weight')
            return list(itertools.islice(paths_generator, k))
        except nx.NetworkXNoPath:
            return []

    def _evaluate_single_leg(self, src, dest, current_clock, user_ftl, user_carting):
        """Handles the AI Dual-Check for a single A -> B jump."""
        road_data = self.osrm_graph[src][dest]
        
        base_payload = {
            'source_hub': src,
            'dest_hub': dest,
            'start_hour': current_clock,
            'osrm_time': road_data['weight'],
            'osrm_distance': road_data['osrm_distance']
        }

        if user_ftl == 1:
            base_payload['is_ftl'] = 1; base_payload['is_carting'] = 0
            return predict_eta(base_payload)
            
        elif user_carting == 1:
            base_payload['is_ftl'] = 0; base_payload['is_carting'] = 1
            return predict_eta(base_payload)
            
        else:
            base_payload['is_ftl'] = 1; base_payload['is_carting'] = 0
            eta_ftl = predict_eta(base_payload)
            
            base_payload['is_ftl'] = 0; base_payload['is_carting'] = 1
            eta_carting = predict_eta(base_payload)
            
            if eta_ftl is None or eta_carting is None:
                print(f"WARNING: AI failed on leg {src} -> {dest}. Skipping.")
                return float('inf')
            
            return min(eta_ftl, eta_carting)


    def process_route_request(self, user_data):
        """
        The Master Function called by Streamlit.
        user_data dict expects: source_name, destination_name, is_ftl, is_carting, start_hour, blacklist_hubs
        """
        src_id = self._get_hub_number(user_data['source_name'])
        dest_id = self._get_hub_number(user_data['destination_name'])

        if src_id is None or dest_id is None:
            return {"error": "Invalid Source or Destination Name."}

        if user_data['start_hour'] == -1:
            now = datetime.now()
            start_clock = now.hour + (now.minute / 60.0)
        else:
            start_clock = float(user_data['start_hour'])

        top_paths = self._get_candidate_paths(src_id, dest_id, user_data.get('blacklist_hubs', []))
        
        if not top_paths:
            return {"error": "No valid route exists between these hubs (check blacklists)."}

        best_path = None
        best_overall_eta = float('inf')
        path_details = []

        # Evaluate every path
        for path in top_paths:
            current_clock = start_clock
            total_path_eta = 0.0
            
            for i in range(len(path) - 1):
                leg_src = path[i]
                leg_dest = path[i+1]
                
                leg_eta = self._evaluate_single_leg(
                    leg_src, leg_dest, current_clock, 
                    user_data['is_ftl'], user_data['is_carting']
                )
                
                total_path_eta = leg_eta + total_path_eta
                current_clock = (current_clock + leg_eta) % 24
                
            path_details.append({"path": path, "total_eta": total_path_eta})

            if total_path_eta < best_overall_eta:
                best_overall_eta = total_path_eta
                best_path = path

        return {
            "success": True,
            "best_path": best_path,
            "best_eta_hours": round(best_overall_eta, 2),
            "all_evaluated_paths": path_details
        }