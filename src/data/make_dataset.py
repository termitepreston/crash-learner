import os
from pathlib import Path

import pandas as pd

# Define project base paths
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def convert_excel_to_processed(filename: str, sheet_name="Sheet1"):
    """
    Reads a raw Excel file from data/raw and saves a CSV version to data/processed.
    """
    input_path = RAW_DATA_DIR / filename
    # Create intermediate csv filename
    output_filename = filename.rsplit(".", 1)[0] + ".csv"
    output_path = PROCESSED_DATA_DIR / output_filename

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print(f"[INFO] Loading raw data from: {input_path} (Sheet: {sheet_name})")

    try:
        # Load the Excel file
        df = pd.read_excel(input_path, sheet_name=sheet_name, engine="openpyxl")
        print(f"[INFO] Raw data loaded. Shape: {df.shape}")

        # Save as CSV (intermediate step before heavy processing)
        df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Intermediate CSV saved to: {output_path}")
        return output_path

    except FileNotFoundError:
        print(f"[ERROR] The file {filename} was not found in {RAW_DATA_DIR}.")
        raise
    except Exception as e:
        print(f"[ERROR] An error occurred during conversion: {e}")
        raise
