
# Quick Start

### Prerequisites

- Python 3.10 or higher

### Installation

1. Clone the repository
   ```bash
   cd appa-chinquinaria
   ```

2. Create and activate a virtual environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file (required for proprietary LLM use)

   If you plan to use a proprietary LLM (GPT-4.1), you must provide an access token. In the root directory of the project, create a file named `.env` containing:
   ```
   GITHUB_TOKEN=your_personal_access_token_here
   ```

## Usage

### Execute complete pipeline
To execute the complete pipeline, set your configuration in `chinquinaria/config.py` and run:
```bash
python -m chinquinaria.pipeline
```

# Chinquinaria Pipeline

This directory contains the automated pipeline for analyzing the importance of environmental variables using baseline models, automated SHAP execution, and report generation via LLM. The project is an integral part of the root project, focused on studying the factors that influence air quality.

## Pipeline Structure

The pipeline consists of the following main modules:

- **data_loading/**: Data loading and pre-processing.
- **modeling/**: Implementation of baseline models (XGBoost, LightGBM, MLP, Random Forest, LSTM) and training/prediction functions.
- **explainability/**: Feature importance calculation using SHAP.
- **llm_reporting/**: Automated textual report generation via LLM.
- **utils/**: Support functions (logging, evaluation, file I/O).
- **pipeline.py**: End-to-end orchestration script.

## Objectives

- Evaluate the importance of environmental variables using baseline models.
- Automate SHAP analysis for interpretability.
- Generate textual reports via LLM for synthesis and communication of results.

## Execution Instructions

The **pipeline executes in sequence**:
     - Data loading and splitting
     - Model training
     - Prediction and metric calculation
     - Automated SHAP analysis
     - LLM report generation
     - Saving results and reports in the output directory
**Output**
   - Results (predictions, feature importance, textual reports) are saved in the directory specified in `config.py`.
   - Log files and intermediate reports are available for detailed analysis.
