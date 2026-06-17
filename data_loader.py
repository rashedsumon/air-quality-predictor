import os
import glob
import pandas as pd
import kagglehub

def load_pollution_data():
    """
    Downloads the Global Urban Air Quality and Pollution Time-Series dataset 
    via kagglehub if not locally available, tracks the path, and returns 
    a consolidated pandas DataFrame.
    """
    print("Downloading/Checking dataset from Kaggle...")
    # Programmatic download utilizing the provided kagglehub framework
    download_path = kagglehub.dataset_download("iconicwasil/global-urban-air-quality-and-pollution-time-series")
    print(f"Path to dataset files: {download_path}")
    
    # Look for CSV files in the downloaded directory path
    csv_files = glob.glob(os.path.join(download_path, "**/*.csv"), recursive=True)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV data found in dataset path: {download_path}")
    
    # Read the main primary dataset file found
    target_csv = csv_files[0]
    print(f"Loading data from: {target_csv}")
    
    df = pd.read_csv(target_csv)
    
    # Clean column names to strip spaces and convert to lowercase for uniform processing
    df.columns = df.columns.str.strip().str.lower()
    
    return df

if __name__ == "__main__":
    # Standard quick sanity check execution
    data = load_pollution_data()
    print("Columns found:", list(data.columns))
    print(data.head(2))