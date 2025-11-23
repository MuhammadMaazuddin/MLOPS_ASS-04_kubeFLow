# MLOps Assignment 4 - Kubeflow Pipelines

This project demonstrates how to build, compile, and orchestrate a Machine Learning pipeline using Kubeflow Pipelines (KFP). It includes components for data preprocessing and model training (Random Forest Regressor), orchestrated to run on a Kubernetes cluster via Minikube.

## Project Description

The pipeline consists of two main components:
1.  **Data Preprocessing**: Reads raw CSV data, cleans it, splits it into training and testing sets, scales features, and saves the processed data.
2.  **Model Training**: Loads the processed training data, trains a Random Forest Regressor, and saves the trained model.

## Prerequisites

-   **OS**: Linux (Ubuntu recommended)
-   **Python**: 3.9+
-   **Minikube**: For running a local Kubernetes cluster.
-   **Kubeflow Pipelines**: Deployed on Minikube.
-   **Docker**: For building component images (if needed, though base images are used here).

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Set up a virtual environment**:
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure `kfp` (v2.x), `pandas`, `scikit-learn`, and `joblib` are installed.*

## Usage

### 1. Compile Components
The pipeline components are defined in `src/pipeline_components.py`. Run this script to generate the component YAML files.

```bash
source env/bin/activate
cd src
python3 pipeline_components.py
```
This will create:
-   `components/data_preprocessing.yaml`
-   `components/model_training.yaml`

### 2. Compile the Pipeline
The pipeline definition is in `pipeline.py`. Run this script to compile the pipeline into a YAML file that can be uploaded to KFP.

```bash
source env/bin/activate
python3 pipeline.py
```
This will create `pipeline.yaml`.

### 3. Run on Kubeflow Pipelines
1.  Start Minikube and ensure KFP is running.
    ```bash
    minikube start
    # Forward port if needed to access KFP dashboard
    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
    ```
2.  Open the KFP Dashboard (usually at `http://localhost:8080`).
3.  Click **Upload Pipeline** and select the generated `pipeline.yaml`.
4.  Create a **Run**.
    -   **Input Data Path**: Ensure the data path provided (default: `/tmp/data/city_A.csv`) is accessible to the pipeline components. You may need to use a Minio URL or mount a volume if running locally on Minikube.

## Directory Structure

```
MLOPS_ASS#04/
├── components/             # Generated component YAML files
│   ├── data_preprocessing.yaml
│   └── model_training.yaml
├── data/                   # Data files
│   └── city_A.csv
├── env/                    # Virtual environment
├── src/                    # Source code
│   ├── __init__.py
│   ├── pipeline_components.py  # Component definitions
│   └── model_training.py       # (Optional) Standalone training script
├── pipeline.py             # Pipeline definition and compilation
├── pipeline.yaml           # Compiled pipeline
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Troubleshooting

-   **KFP Version**: This project uses KFP v2 syntax (`kfp.compiler.Compiler`). Ensure your environment matches.
-   **Data Access**: If the pipeline fails to find the input file, check if the path is accessible from within the Kubernetes pods. Using an object store like Minio is recommended for KFP.
