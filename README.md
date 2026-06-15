# 🚚 Intelligent Logistics Routing & ETA Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://delhivery-eta.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

An end-to-end AI logistics routing dashboard built to optimize complex supply chain networks. This project combines **Graph Theory** for pathfinding with **Deep Learning (Graph Neural Networks)** to predict real-world Estimated Time of Arrival (ETA) by analyzing historical traffic, delays, and vehicle types.

**🔴 Live Demo:** [https://delhivery-eta.streamlit.app/](https://delhivery-eta.streamlit.app/)

---

## ✨ Key Features

* **Intelligent Path Generation:** Utilizes NetworkX and Dijkstra's algorithm to compute physically valid network paths between hubs.
* **Deep Learning ETA Prediction:** A PyTorch Geometric Graph Neural Network (GNN) evaluates every leg of the journey, adjusting for real-world delays.
* **Dynamic Time Simulation:** A "Rolling Clock" system tracks the exact hour of arrival at intermediate hubs to accurately predict subsequent traffic conditions.
* **Vehicle Optimization:** Supports Full Truckload (FTL) and Carting routing, plus an **Auto-Optimize** mode that dynamically switches vehicle types leg-by-leg to find the absolute fastest route.
* **Live Hub Analytics:** Real-time metrics on hub bottleneck scores, betweenness centrality, and historical delay ratios.
* **Dynamic Constraint Engine:** Allows operators to blacklist specific hubs (e.g., due to weather or maintenance) and instantly recalculates the network.

---

## 🏗️ System Architecture

This engine operates in three seamless phases:

1.  **Phase 1: Candidate Path Generation (NetworkX)**
    * Loads an OSRM (Open Source Routing Machine) topological graph into memory.
    * Prunes blacklisted hubs on the fly using a lightweight temporary graph.
    * Generates the top *K* shortest physical paths.
2.  **Phase 2: The AI Evaluation Engine (PyTorch)**
    * Chops candidate paths into independent legs using a sliding window.
    * Feeds node embeddings and edge features (`is_ftl`, `start_hour`, `osrm_time`, `osrm_distance`) into a pre-trained `DelhiveryGNN` model.
    * Pushes the clock forward automatically using modulo arithmetic (`% 24`) to ensure temporal accuracy.
3.  **Phase 3: Interactive Dashboard (Streamlit)**
    * Provides a responsive, data-driven UI.
    * Instantly maps integer-based model outputs back to human-readable Hub Names.
    * Displays alternative routes and real-time operational warnings for bottlenecked hubs.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **AI/ML Backend:** PyTorch, PyTorch Geometric, Scikit-Learn
* **Graph Processing:** NetworkX
* **Data Manipulation:** Pandas, NumPy, Joblib

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. We use [uv](https://github.com/astral-sh/uv) for blazing-fast dependency management.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/swan556/ETA-pred-IITG.git
   cd ETA-pred-IITG
   ```

2. **Create a virtual environment and install dependencies:**
   *(Ensure you install the correct version of PyTorch and PyTorch Geometric for your system/CUDA setup)*
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

### Running the Application

With `uv`, you can run the Streamlit dashboard directly:

```bash
uv run streamlit run src/interface/interface.py
```

The application will spin up a local server, usually accessible at `http://localhost:8501`.

---

## 📂 Project Structure

```text
├── dataset/                  # Precomputed NetworkX graphs, scalers, and node mappings
│   ├── osrm_networkx_graph.pkl
│   ├── hub-features.csv
│   └── ...
├── final_model/              # Saved PyTorch GNN weights
│   └── delhivery_gnn_weights.pth
├── src/
│   └── interface/            
│       ├── interface.py      # Streamlit UI Dashboard
│       ├── engine.py         # RoutingEngine class bridging NetworkX and PyTorch
│       └── runner.py         # AI Inference script (predict_eta)
├── README.md
└── requirements.txt
```

---

## 👨‍💻 Developer

Developed as an end-to-end machine learning system design project, integrating competitive programming logic with production-ready AI deployment.
