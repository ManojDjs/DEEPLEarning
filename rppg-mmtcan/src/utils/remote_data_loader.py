import os
import requests
from zipfile import ZipFile
from io import BytesIO

def download_dataset_from_github(repo_url, file_path, local_dir):
    """
    Download and extract a dataset from a public GitHub repo URL.
    
    Args:
    - repo_url: URL of the raw dataset file in the GitHub repository
    - file_path: Relative file path to the dataset in the repo
    - local_dir: Local directory to save the downloaded dataset
    """
    os.makedirs(local_dir, exist_ok=True)
    
    # Construct the URL for the raw file
    raw_url = repo_url + "/raw/main/" + file_path
    
    # Send HTTP request to get the dataset file
    response = requests.get(raw_url)
    
    if response.status_code == 200:
        # Assuming it's a zip file, you can adapt this part to handle other formats (e.g., .tar)
        with ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(local_dir)
            print(f"Dataset downloaded and extracted to {local_dir}")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")

# Example usage
repo_url = "https://github.com/AiPEX-Lab/rppg_biases.git"
file_path = "https://github.com/AiPEX-Lab/rppg_biases.git/Data"  # Replace with the actual path in the GitHub repo
local_dir = "Data/SampleData"  # Local directory to save data

download_dataset_from_github(repo_url, file_path, local_dir)
