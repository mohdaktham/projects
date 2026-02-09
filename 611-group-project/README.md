# MGTA 611 Group Project — Deep Learning for Retail Sales Forecasting

This repository contains the **MGTA 611 Term Project** (W2026): a deep learning and machine learning pipeline for retail sales prediction using the Walmart dataset (train, test, features, stores).

## Contents

- **611 Script.ipynb** — Main Jupyter notebook with:
  - Data loading and preparation
  - Advanced feature engineering (holidays, rolling stats, etc.)
  - XGBoost baseline with custom holiday-weighted loss
  - LSTM model (TensorFlow/Keras) for time-series forecasting
  - LSTM tuning (callbacks, hyperparameters)
  - SHAP-based model interpretation
- **Data/** — CSV datasets:
  - `train.csv` — Historical weekly sales (Store, Dept, Date, Weekly_Sales, IsHoliday)
  - `test.csv` — Test period (Store, Dept, Date, IsHoliday)
  - `features.csv` — Store-date features (Temperature, Fuel_Price, MarkDowns, CPI, Unemployment, etc.)
  - `stores.csv` — Store metadata (Type, Size)
- **MGTA 611 Term Project Proposal.docx** — Project proposal
- **MGTA611-W2026-Deep Learning Project.pdf** — Course project description
- **Planning Document.docx** / **Project Planning_.docx** — Planning notes

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/611-group-project.git
cd 611-group-project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

Open **611 Script.ipynb** in Jupyter Lab, Jupyter Notebook, or VS Code. The notebook is configured to read data from the **Data/** folder, so run it from the **project root** (this directory).

- **Jupyter:** `jupyter notebook` or `jupyter lab`
- **VS Code:** Open the notebook and select the kernel (e.g. the venv you created)

## Requirements

- Python 3.8+
- See **requirements.txt** for packages: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, tensorflow, shap

## Project structure

```
611-group-project/
├── 611 Script.ipynb          # Main analysis & models
├── Data/
│   ├── train.csv
│   ├── test.csv
│   ├── features.csv
│   └── stores.csv
├── README.md                  # This file
├── requirements.txt
├── .gitignore
└── docs/                      # Proposal & planning (optional)
    └── MGTA 611 Term Project Proposal.docx
```

## Citation / Course

MGTA 611 — Deep Learning Project (W2026).  
Data and project description as provided in the course materials.
