# pip install virtualenv
# python -m virtualenv deepseek_env
# deepseek_env\Scripts\activate

import os
import pandas as pd
import ollama
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np


# Model configuration
desired_model = 'deepseek-r1:14b'

# Target columns for intelligent merging
TARGET_COLUMNS = ["Mobile", "Email", "Name", "City", "State","Pincode"]
root_directories = ["A","B","C","D","E","F","G","H","I"]


# Initialize Sentence-BERT model for semantic similarity
semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def read_csv_files(root_directory):
    csv_files = []
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(subdir, file))
    return csv_files


def calculate_similarity(source_list, target_list):
    source_embeddings = semantic_model.encode(source_list)
    target_embeddings = semantic_model.encode(target_list)
    similarities = []
    for src_emb in source_embeddings:
        similarity_scores = np.dot(target_embeddings, src_emb) / (np.linalg.norm(target_embeddings, axis=1) * np.linalg.norm(src_emb))
        similarities.append(similarity_scores)
    return similarities


def get_header_similarity(headers):
    prompt = f"Match the following headers to the closest ones from this list: {TARGET_COLUMNS}. Return a dictionary where keys are from the target list and values are the closest matches: {headers}"
    
    try:
        response = ollama.chat(model=desired_model, messages=[{'role': 'user', 'content': prompt}])
        ollama_response = response.get('message', {}).get('content', 'No response content')
        print("Model Response:", ollama_response)
        
        try:
            mapped_headers = eval(ollama_response)
        except Exception as e:
            print("Error parsing model response:", e)
            mapped_headers = {col: [] for col in TARGET_COLUMNS}
    except Exception as e:
        print(f"Error in DeepSeek request: {e}")
        mapped_headers = {col: [] for col in TARGET_COLUMNS}
    
    similarities = calculate_similarity(headers, TARGET_COLUMNS)
    
    semantic_mapped_headers = {target_col: [] for target_col in TARGET_COLUMNS}
    for idx, similarity_scores in enumerate(similarities):
        best_match_idx = np.argmax(similarity_scores)
        semantic_mapped_headers[TARGET_COLUMNS[best_match_idx]].append(headers[idx])
    
    for target_col in semantic_mapped_headers:
        if not semantic_mapped_headers[target_col]:
            semantic_mapped_headers[target_col] = mapped_headers.get(target_col, [])
    
    return semantic_mapped_headers


def is_valid_mobile(value):
    return str(value).isdigit() and 7 <= len(str(value)) <= 15


def merge_dataframes(file_paths):
    dataframes = []
    header_sets = set()

    for file_path in tqdm(file_paths, desc="Reading CSV files"):
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
            header_sets.update(df.columns.tolist())
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    header_mapping = get_header_similarity(list(header_sets))

    merged_frames = []

    for df in tqdm(dataframes, desc="Merging DataFrames intelligently"):
        merged_dict = {target_col: pd.Series(dtype='object') for target_col in TARGET_COLUMNS}

        for target_col, mapped_cols in header_mapping.items():
            column_data = pd.Series(dtype='object')

            for col in mapped_cols:
                if col in df.columns:
                    clean_data = df[col].dropna()

                    # Apply additional filters for specific columns
                    if target_col == "Mobile":
                        clean_data = clean_data[clean_data.apply(is_valid_mobile)]  # Ensure valid mobile numbers
                    elif target_col == "Pincode":
                        clean_data = clean_data[clean_data.apply(lambda x: str(x).isdigit() and len(str(x)) == 6)]

                    column_data = column_data.combine_first(clean_data)

            merged_dict[target_col] = column_data

        merged_frames.append(pd.DataFrame(merged_dict))

    final_df = pd.concat(merged_frames, ignore_index=True)
    return final_df


def save_dataframe_in_parts(df, base_filename, max_rows_per_file=500000):
    num_parts = (len(df) // max_rows_per_file) + 1
    for part in range(num_parts):
        start_idx = part * max_rows_per_file
        end_idx = min((part + 1) * max_rows_per_file, len(df))
        part_df = df.iloc[start_idx:end_idx]

        if not part_df.empty:
            output_file = f"{base_filename}_part_{part + 1}.csv"
            part_df.to_csv(output_file, index=False)
            print(f"Saved {output_file} with {len(part_df)} rows.")


def main():
    for root_directory in root_directories:
        directory_name = os.path.basename(root_directory).replace(' ', '_')
        print(f"Processing directory: {root_directory}")
        csv_files = read_csv_files(root_directory)
        if not csv_files:
            print(f"No CSV files found in {root_directory}.")
            continue

        print(f"Found {len(csv_files)} CSV files in {root_directory}.")

        merged_df = merge_dataframes(csv_files)

        base_filename = f"{directory_name}"
        save_dataframe_in_parts(merged_df, base_filename)


if __name__ == "__main__":
    main()
