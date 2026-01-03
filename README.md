# Neural Network-Based Classification of Road Traffic Accident Severity

## Project Description

This project implements a Neural Network-based classification model to predict the severity of road traffic accidents. Utilizing a dataset primarily from Addis Ababa, the project covers the full machine learning workflow: from extensive Exploratory Data Analysis (EDA) and robust data preprocessing to neural network model design, training, and comprehensive evaluation. The goal is to build a model capable of classifying accident types into categories such as 'Fatal', 'Serious Injury', 'Minor Injury', and 'Property Damage Only (PDO)'.

The repository is structured to promote modularity and reproducibility, with dedicated folders for raw/processed data, Jupyter notebooks for experimentation, Python source code for core functionalities (data handling, model architecture, training, evaluation), trained models, and reports.

## Getting Started

Follow these steps to set up the project environment and run the analyses.

### 1. Prerequisites

Ensure you have Python 3.9+ installed.

### 2. Install `uv` (Recommended Python Package Manager)

If you don't have `uv` installed, it's a fast and modern alternative to `pip` and `venv`.

```bash
pip install uv
```

### 3. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/termitepreston/crash-learner.git
cd crash-learner
```

### 4. Create and Activate Virtual Environment

It's highly recommended to work within a virtual environment.

```bash
# Create virtual environment
uv venv

# Activate virtual environment (Linux/macOS)
source .venv/bin/activate
```

```bat
# Activate virtual environment (Windows Command Prompt)
.venv\Scripts\activate.bat

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

### 5. Install Python Dependencies

With your virtual environment activated, install all project dependencies using `uv`:

```bash
uv sync
```

This command reads `pyproject.toml` and installs all necessary packages.

### 6. Install Graphviz (System-wide for Model Visualization)

Graphviz is required by `tf.keras.utils.plot_model` to visualize the neural network architecture.

- Linux (Debian/Ubuntu):

```bash
sudo apt-get update
sudo apt-get install graphviz
```

- Windows

1. Download the installer from the official [Graphviz website](https://graphviz.org/download/).
2. During installation, ensure you select the option to "Add Graphviz to the system PATH for all users". If not, you'll need to manually add the `bin` directory of your Graphviz installation to your system's PATH environment variable.

### 7. Install Typst (System-wide for Report Generation)

Typst is a new markup-based typesetting system used for generating the project report. For installation, follow the instructions on the official Typst website.
  
- Verify Installation:

```bash
typst --version
```

- Required Fonts for Typst:

The report (`reports/report.typ`) uses specific fonts. For Typst to render correctly, these fonts must be available on your system. Please ensure the following fonts are installed on your operating system:

- `STIX Two Text`
- `STIX Two Math`
- `CMU Typewriter Text`
- `Cooper`
- `CMU Sans Serif`
- `Inter Display`

You can verify which fonts Typst can find using:

```bash
typst fonts
```

## Project Structure

```text
my-ml-project/
├── data/                     # Raw and processed data
│   ├── raw/                  # Original, immutable data dumps (e.g., crash-data-aa.xlsx)
│   └── processed/            # Cleaned, transformed data ready for modeling (e.g., X_train.parquet)
│
├── notebooks/                # Jupyter notebooks for exploration & prototyping
│   ├── 01-data-exploration.ipynb
│   ├── 02-feature-engineering.ipynb
│   ├── 03-model-experiments.ipynb
│   ├── 04-training-process.ipynb
│   └── 05-evaluation.ipynb
│
├── src/                      # Source code (Python package)
│   ├── data/                 # Data ingestion & preprocessing
│   │   ├── make_dataset.py   # Script to convert raw data (e.g., Excel to CSV)
│   │   └── preprocess.py     # Cleaning, feature engineering, data splitting, scaling, encoding
│   │
│   └── models/               # Model training & prediction
│       ├── model_arch.py     # Neural network architecture definition
│       ├── train.py          # Training pipeline with callbacks and class weights
│       └── evaluate.py       # Model scoring & metrics
│
├── models/                   # Trained model artifacts (e.g., best_model.keras)
│
├── reports/                  # Analysis reports, figures, metrics
│   ├── figures/              # Saved plots (e.g., confusion matrices, training curves)
│   └── report.typ            # Source file for the final project report
│
├── pyproject.toml            # Project metadata & dependencies
└── README.md                 # Project overview, setup instructions
```

## Usage

### 1. Run Jupyter Lab

Navigate to the project root directory and launch Jupyter Lab:

```bash
jupyter lab
```

Then, sequentially run the notebooks `01-data-exploration.ipynb` through `05-evaluation.ipynb` in the `notebooks/` directory. Each notebook builds upon the previous one, performing distinct tasks of the machine learning pipeline.

### 2. Generate the Project Report

After completing the notebooks (which generate all necessary figures in `reports/figures/`), you can compile the `report.typ` file into a PDF.

Navigate to the `reports/` directory:

```bash
cd reports/
```

Then, compile the report using Typst:

```bash
typst compile report.typ
```

This will generate `report.pdf` in the `reports/` directory.
