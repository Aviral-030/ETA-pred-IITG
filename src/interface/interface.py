import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv
import numpy as np
import joblib

# ==========================================
# 1. THE MODEL BLUEPRINT
# (Required so PyTorch knows where to put the weights)
# ==========================================
class DelhiveryGNN(nn.Module):
    def __init__(self, num_nodes, embed_dim=16, num_edge_features=6):
        super(DelhiveryGNN, self).__init__()
        self.node_emb = nn.Embedding(num_nodes, embed_dim)
        self.conv1 = CGConv(channels=embed_dim, dim=num_edge_features)
        
        predictor_input_size = (embed_dim * 2) + num_edge_features
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_size, 32),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) 
        )

    def forward(self, edge_index, edge_attr):
        pass # Not needed for direct single-edge evaluation

# ==========================================
# 2. LOAD THE ASSETS INTO MEMORY
# ==========================================
# Make sure these paths point to where your files actually are
try:
    scaler = joblib.load('dataset/fitted_scaler.pkl')
    node_mapping = joblib.load('dataset/node_mapping.pkl')
    
    model = DelhiveryGNN(num_nodes=len(node_mapping), embed_dim=16, num_edge_features=6)
    model.load_state_dict(torch.load("final_model/delhivery_gnn_weights.pth"))
    model.eval() # Lock the model for inference
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Missing asset file. {e}")
    exit()

# ==========================================
# 3. THE PARSER & PREDICTOR FUNCTION
# ==========================================
def predict_from_string(data_string):
    # Parse the comma-separated string
    data_parts = data_string.split(',')
    
    try:
        source_hub_raw = float(data_parts[1])
        dest_hub_raw = float(data_parts[2])
        is_carting = float(data_parts[3])
        is_ftl = float(data_parts[4])
        day_of_week = float(data_parts[5])
        start_hour = float(data_parts[6])
        raw_osrm_time = float(data_parts[10])
        raw_osrm_distance = float(data_parts[11])
        actual_time = float(data_parts[13]) # Extracted just for comparison
    except IndexError:
        return "ERROR: String format is invalid or missing columns."

    print(f"\n--- PREDICTING ETA: Hub {int(source_hub_raw)} -> Hub {int(dest_hub_raw)} ---")
    
    # Check if hubs exist in our trained universe
    if source_hub_raw not in node_mapping or dest_hub_raw not in node_mapping:
        print("ERROR: One of these hubs is brand new and wasn't in the training data!")
        return None

    # Map to internal IDs
    src_id = torch.tensor([node_mapping[source_hub_raw]], dtype=torch.long)
    dst_id = torch.tensor([node_mapping[dest_hub_raw]], dtype=torch.long)
    
    # Scale the continuous features (requires a 4-item array to match scaler shape)
    dummy_input = np.array([[raw_osrm_time, raw_osrm_distance, 0.0, 0.0]])
    scaled_values = scaler.transform(dummy_input)[0]
    
    # Build the final attribute tensor
    edge_attr = torch.tensor([[is_carting, is_ftl, day_of_week, start_hour, scaled_values[0], scaled_values[1]]], dtype=torch.float32)
    
    # Run the math
    with torch.no_grad():
        x_src = model.node_emb(src_id)
        x_dst = model.node_emb(dst_id)
        regression_input = torch.cat([x_src, x_dst, edge_attr], dim=1)
        predicted_factor = model.predictor(regression_input).item()
        
    # Apply bounds and calculate final time
    predicted_factor = min(predicted_factor, 4.0)
    true_eta = raw_osrm_time * predicted_factor
    
    # Output the results
    print(f"OSRM Base Time:       {raw_osrm_time:.2f} hours")
    print(f"AI Predicted Factor:  {predicted_factor:.2f}x")
    print(f"FINAL AI ETA:         {true_eta:.2f} hours")
    print(f"(Actual target time was {actual_time:.2f} hours)")
    print("-" * 50)
    
    return true_eta

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    testing_str = "test,444,30,1,0,3,23,0,1,0,52.0,56.2137,45.90293875782862,68.0,1.3076923076923077"
    predict_from_string(testing_str)