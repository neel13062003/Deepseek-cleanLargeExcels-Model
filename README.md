# CSV Data Processing & Merging Pipeline

## Overview
This project processes and merges multiple CSV datasets by intelligently mapping column headers, cleaning data, and ensuring consistency. It leverages AI-based semantic similarity to standardize headers and applies filtering rules for valid data entries.

## Features
- **Recursive CSV File Search:** Collects all CSV files from a specified directory.
- **AI-Powered Header Mapping:** Uses Sentence-BERT (`paraphrase-MiniLM-L6-v2`) and DeepSeek-R1:14B to map headers to predefined target columns.
- **Data Cleaning & Validation:**
  - Filters valid phone numbers (7 to 15 digits).
  - Normalizes column names for consistency.
- **Merging & Structuring:** Combines multiple datasets into a structured format.

## Installation

### Step 1: Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### Step 2: Install Dependencies
```sh
pip install -r requirements.txt
```

### Step 3: Ensure Required Models are Installed
```sh
python -m sentence_transformers.sentence_transformer
```

## Usage

### Running the Script
```sh
python main.py --root_directory path/to/csv/files
```

### Arguments
- `--root_directory`: Path to the folder containing CSV files.

## Project Structure
```
project/
│── main.py                # Main script
│── utils.py               # Utility functions
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
└── data/                  # Folder for CSV files
```

## Functions

### `read_csv_files(root_directory)`
Recursively scans a directory to find all CSV files.

### `calculate_similarity(source_list, target_list)`
Computes cosine similarity between dataset headers and target headers.

### `get_header_similarity(headers)`
Uses DeepSeek AI to match dataset headers to predefined target columns.

### `is_valid_mobile(value)`
Checks if a given value is a valid phone number.

### `merge_dataframes(file_paths)`
Reads, maps, cleans, and merges CSV data.

## Future Enhancements
- **Parallel Processing:** Implement multiprocessing for handling large datasets.
- **Improved Header Mapping:** Train a custom model for better accuracy.
- **Batch Processing:** Process CSV files in chunks to optimize memory usage.

## License
This project is licensed under the MIT License.

---
**NEEL N SONI:**
