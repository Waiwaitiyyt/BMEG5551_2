# This is a script to automatically download the dataset from kaggle
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
os.environ["KAGGLEHUB_CACHE"] = str(root_dir)  

import kagglehub
import matplotlib.pyplot as plt


path = kagglehub.dataset_download("tawsifurrahman/aseptic-loose-hip-implant-xray-database")

print("Path to dataset files:", path)

# Dataset are stored in the root dir defaultly.