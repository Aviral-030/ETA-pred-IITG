import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv
import numpy as np
import joblib

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
    scaler = joblib.load('dataset/fitted_scaler.pkl')
    node_mapping = joblib.load('dataset/node_mapping.pkl')
    
    model = DelhiveryGNN(num_nodes=len(node_mapping), embed_dim=16, num_edge_features=5)
    model.load_state_dict(torch.load("final_model/delhivery_gnn_weights.pth"))
    model.eval() 
    
    print("Reconstructing supply chain network...")
    df = pd.read_csv("dataset/final_normalized_graph.csv")
    
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

def predict_from_string(data):
    # data_parts = data_string.split(',')
    
    try:
        source_hub_raw = float(data['source_hub'])
        dest_hub_raw = float(data['destination_hub'])
        is_carting = float(data['is_carting'])
        is_ftl = float(data['is_ftl'])
        start_hour = float(data['start_hour'])
        raw_osrm_time = float(data['osrm_time'])
    except Exception as e:
        print(e)
        

    print(f"\n--- PREDICTING ETA: Hub {int(source_hub_raw)} -> Hub {int(dest_hub_raw)} ---")
    
    if source_hub_raw not in node_mapping or dest_hub_raw not in node_mapping:
        print("ERROR: One of these hubs is brand new and wasn't in the training data!")
        return None

    src_idx = node_mapping[source_hub_raw]
    dst_idx = node_mapping[dest_hub_raw]
    
    x_src = global_context_embeddings[src_idx].unsqueeze(0)
    x_dst = global_context_embeddings[dst_idx].unsqueeze(0)
    
    dummy_input = np.array([[raw_osrm_time, 0.0, 0.0]])
    scaled_values = scaler.transform(dummy_input)[0]
    
    edge_attr = torch.tensor([[is_carting, is_ftl, start_hour, scaled_values[0], scaled_values[1]]], dtype=torch.float32)
    
    with torch.no_grad():
        regression_input = torch.cat([x_src, x_dst, edge_attr], dim=1)
        predicted_factor = model.predictor(regression_input).item()
        
    predicted_factor = min(predicted_factor, 4.0)
    true_eta = raw_osrm_time * predicted_factor
    
    print(f"OSRM Base Time:       {raw_osrm_time:.2f} hours")
    print(f"AI Predicted Factor:  {predicted_factor:.2f}x")
    print(f"FINAL AI ETA:         {true_eta:.2f} hours")
    print("-" * 50)
    
    return true_eta

if __name__ == "__main__":
    # while(1):
    #     testing_str = input(">>>")
    #     if(testing_str=="q"): break
    #     predict_from_string(testing_str)
    ...