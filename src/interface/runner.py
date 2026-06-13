import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv
import numpy as np
import joblib
import os

class DelhiveryGNN(nn.Module):
    def __init__(self, num_nodes, embed_dim=16, num_edge_features=5):
        super(DelhiveryGNN, self).__init__()
        
        self.node_emb = nn.Embedding(num_nodes, embed_dim)
        
        self.conv1 = CGConv(channels=embed_dim, dim=num_edge_features)
        self.conv2 = CGConv(channels=embed_dim, dim=num_edge_features)
        self.conv3 = CGConv(channels=embed_dim, dim=num_edge_features)

        predictor_input_size = (embed_dim * 2) + num_edge_features
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_size, 64), 
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 1) 
        )

    def get_contextual_embeddings(self, edge_index, edge_attr):
        num_nodes = self.node_emb.num_embeddings
        node_ids = torch.arange(num_nodes, device=edge_index.device)
        x = self.node_emb(node_ids)
        
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        
        return x

try:
    print("Booting AI Model and loading assets...")
    current_dir = os.path.dirname(os.path.abspath(__file__))    
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))

    fitted_scale_path = os.path.join(project_root, 'dataset', 'fitted_scaler.pkl')
    node_mapping_path = os.path.join(project_root, 'dataset', 'node_mapping.pkl')
    final_model_weights_path = os.path.join(project_root, 'final_model', 'delhivery_gnn_weights.pth')

    scaler = joblib.load(fitted_scale_path)
    node_mapping = joblib.load(node_mapping_path)

    model = DelhiveryGNN(num_nodes=len(node_mapping), embed_dim=16, num_edge_features=5)
    model.load_state_dict(torch.load(final_model_weights_path))
    model.eval() 
    
    print("Reconstructing supply chain network...")
    df = pd.read_csv(os.path.join(project_root, 'dataset', 'final_normalized_graph.csv'))
    
    src = df['source_number'].map(node_mapping).values
    dst = df['destination_number'].map(node_mapping).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    feature_cols = ['is_carting', 'is_ftl', 'start_hour', 'osrm_time', 'osrm_distance']
    edge_attr = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    
    with torch.no_grad():
        global_context_embeddings = model.get_contextual_embeddings(edge_index, edge_attr)
        
    print("System Ready. Global embeddings precomputed.")

except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Missing asset file. {e}")
    exit()


def predict_eta(data):
    """
    Takes a dictionary payload from the RoutingEngine and returns the ETA in hours.
    """
    try:
        # 1. Safely extract, checking for both naming conventions just in case
        source_hub_raw = data['source_hub']
        dest_hub_raw = data.get('destination_hub', data.get('dest_hub'))
        
        is_carting = float(data['is_carting'])
        is_ftl = float(data['is_ftl'])
        start_hour = float(data['start_hour'])
        raw_osrm_time = float(data['osrm_time'])
        raw_osrm_distance = float(data['osrm_distance'])

    except Exception as e:
        print(f"\n[❌ GNN ERROR] Data Extraction Failed: {e}")
        print(f"Payload received from engine: {data}")
        return None

    # 2. Fuzzy dictionary lookup helper to bypass int/float/string mismatches
    def get_mapped_idx(hub_val):
        if hub_val in node_mapping: return node_mapping[hub_val]
        try:
            if float(hub_val) in node_mapping: return node_mapping[float(hub_val)]
            if int(hub_val) in node_mapping: return node_mapping[int(hub_val)]
        except: pass
        if str(hub_val) in node_mapping: return node_mapping[str(hub_val)]
        return None

    # 3. Safely map the nodes
    src_idx = get_mapped_idx(source_hub_raw)
    dst_idx = get_mapped_idx(dest_hub_raw)
    
    if src_idx is None or dst_idx is None:
        print(f"\n[❌ GNN ERROR] Unmapped Hubs! Src: {source_hub_raw} (Mapped: {src_idx}), Dest: {dest_hub_raw} (Mapped: {dst_idx})")
        return None

    # 4. AI Inference
    x_src = global_context_embeddings[src_idx].unsqueeze(0)
    x_dst = global_context_embeddings[dst_idx].unsqueeze(0)
    
    # 4 columns to perfectly match your fitted scaler!
    dummy_input = np.array([[raw_osrm_time, raw_osrm_distance, 0.0, 0.0]])
    scaled_values = scaler.transform(dummy_input)[0]
    
    edge_attr = torch.tensor([[is_carting, is_ftl, start_hour, scaled_values[0], scaled_values[1]]], dtype=torch.float32)
    
    with torch.no_grad():
        regression_input = torch.cat([x_src, x_dst, edge_attr], dim=1)
        predicted_factor = model.predictor(regression_input).item()
        
    predicted_factor = min(predicted_factor, 4.0)
    true_eta = raw_osrm_time * predicted_factor
    
    return true_eta
