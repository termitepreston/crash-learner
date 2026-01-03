import os
from pathlib import Path

import pandas as pd

# Define project base paths
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def convert_excel_to_processed(filename: str, sheet_name=0):
    """
    Reads a raw Excel file from data/raw and saves a CSV version to data/processed.

    Args:
        filename (str): Name of the excel file (e.g., 'RTA_Dataset.xlsx')
        sheet_name (int/str): The sheet to read. Defaults to the first sheet.
    """
    input_path = RAW_DATA_DIR / filename
    output_filename = filename.replace(".xlsx", ".csv").replace(".xls", ".csv")
    output_path = PROCESSED_DATA_DIR / output_filename

    # Ensure processed directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print(f"[INFO] Loading raw data from: {input_path}")

    try:
        # Load the Excel file
        # engine='openpyxl' is required for .xlsx files
        df = pd.read_excel(input_path, sheet_name=sheet_name, engine="openpyxl")

        # Basic check to ensure data loaded
        print(f"[INFO] Raw data loaded. Shape: {df.shape}")

        # Save as CSV to processed folder
        # index=False prevents pandas from adding an extra index column
        df.to_csv(output_path, index=False)

        print(f"[SUCCESS] Data converted and saved to: {output_path}")
        return output_path

    except FileNotFoundError:
        print(f"[ERROR] The file {filename} was not found in {RAW_DATA_DIR}.")
    except Exception as e:
        print(f"[ERROR] An error occurred during conversion: {e}")


if __name__ == "__main__":
    # Example usage:
    # Replace 'RTA_Dataset.xlsx' with your actual raw excel filename
    # If you only have the sample csv for now, you can skip running this
    # or place your .xlsx file in data/raw/

    # Assuming the raw file might be named 'Road_Traffic_Accident.xlsx' based on the dataset content
    raw_file_name = "crash-data-aa.xlsx"

    # Check if raw dir exists before running
    if os.path.exists(RAW_DATA_DIR):
        convert_excel_to_processed(raw_file_name)
    else:
        print(
            f"[WARNING] Raw directory {RAW_DATA_DIR} does not exist. Please create it."
        )
