# Daic-Woz-LSTM-Graph

Official implementation for multimodal depression detection using BiLSTM/GRU and Graph Neural Networks (GNN) on the DAIC-WOZ dataset.

This project integrates text, audio, and visual features from the [DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz)] dataset. It employs a topic-based graph structure to model the temporal and semantic relationships within clinical interviews.

---

## 🚀 Pipeline Overview

The project follows a sequential pipeline from data preparation to advanced analysis:

1.  **Dataset Acquisition**: Download the original DAIC-WOZ dataset.
2.  **Preprocessing**: Clean and format data using `notebooks/data_process.ipynb`.
3.  **Topic Classification**: Utilize LLMs to classify dialogue topics via `notebooks/topic.ipynb`.
4.  **Model Training**: Execute experiments using BiLSTM (`graph/`) or GRU (`graph_GRU/`) modules.
5.  **Hyperparameter Optimization**: Conduct experiments with [Optuna](https://optuna.org/) using `optuna_train/`.
6.  **In-depth Analysis**: Evaluate model performance and explainability using `graph_explanation/`.

---

## 🛠️ Requirements & Setup

### Environment
- **Language**: Python 3.10+
- **Frameworks**: PyTorch, PyTorch Geometric (PyG), Optuna, Sentence-Transformers
- **Package Manager**: pip

### Installation
```bash
pip install -r requirements.txt
```
*Note: Ensure you have the appropriate CUDA version installed for PyTorch and PyG compatibility.*

### Configuration (.env)
Create a `.env` file in the root directory and add your OpenAI API key for topic classification:
```text
OPENAI_API_KEY=your_api_key_here
```

---

## 📊 Data Preparation

1.  **Original Dataset**: Ensure the DAIC-WOZ dataset is located in the `data/` directory.
2.  **Preprocessing**: Run `notebooks/data_process.ipynb` to process raw transcripts and multimodal features.
3.  **Topic Labeling**: Run `notebooks/topic.ipynb` to perform LLM-based topic extraction. This step is crucial for the topic-based graph construction.

---

## 🏋️ Training & Experiments

You can train individual modules or run hyperparameter optimization.

### Single Model Training
Run the training script for BiLSTM or GRU modules. 

#### Example: Multimodal Topic BiLSTM Proxy
```bash
python -m graph.multimodal_topic_bilstm_proxy.train --num_epochs 100 --config graph/configs/architecture_TT_GAT.yaml --save_dir checkpoints --save_dir_ topic_bilstm_proxy
```

#### Example: Multimodal Topic GRU Proxy
```bash
python -m graph_GRU.multimodal_topic_gru_proxy.train --num_epochs 100 --config graph_GRU/configs/architecture_TT_GAT.yaml --save_dir checkpoints --save_dir_ topic_gru_proxy
```

### Argument Usage (Parse Args)
Commonly used arguments for proxy modules:
- `--num_epochs`: Number of training epochs (default: 100).
- `--config`: Path to the YAML configuration file.
- `--resume`: Path to a checkpoint to resume training from.
- `--save_dir`: Base directory for saving checkpoints.
- `--save_dir_`: Specific subdirectory for the current run.

### Optuna Optimization
To perform automated hyperparameter search:
- **BiLSTM**: `python optuna_train/optuna_graph.py`
- **GRU**: `python optuna_train/optuna_graph_gru.py`

---

## ⚙️ Configurations

Model architectures and search spaces are managed via YAML files:

| Type | Configuration File | Description |
| :--- | :--- | :--- |
| **BiLSTM Architecture** | `graph/configs/architecture_TT_GAT.yaml` | Standard architecture for LSTM-GNN models. |
| **GRU Architecture** | `graph_GRU/configs/architecture_TT_GAT.yaml` | Standard architecture for GRU-GNN models. |
| **Optuna (BiLSTM)** | `optuna_train/optuna_search_grid.yaml` | Search space for BiLSTM optimization. |
| **Optuna (GRU)** | `optuna_train/optuna_search_grid_gru.yaml` | Search space for GRU optimization. |

---

## 🔍 Analysis & Explainability (`graph_explanation/`)

For deep analysis of the models:

- **F1 Score Comparison**: Use `graph_explanation/f1_visualization.py` (or `.ipynb`) to compare F1 scores across various Optuna-trained models.
  ```bash
  python graph_explanation/f1_visualization.py --model_dir checkpoints_optuna
  ```
- **GNN Explainer**: Use `graph_explanation/visualization_audio_video_text.ipynb` to perform in-depth analysis using GNNExplainer, visualizing the importance of audio, video, and text features within the graph.

---

## 📂 Project Structure

```text
.
├── graph/                # BiLSTM-based GNN models
│   └── configs/          # YAML configurations for BiLSTM
├── graph_GRU/            # GRU-based GNN models
│   └── configs/          # YAML configurations for GRU
├── graph_explanation/    # Visualization and explainability tools
├── notebooks/            # Data processing and topic classification (Jupyter)
├── optuna_train/         # Optuna hyperparameter optimization scripts
├── data/                 # Dataset storage (DAIC-WOZ)
├── checkpoints/          # Model checkpoints
└── requirements.txt      # Dependency list
```