import os
import pandas as pd
from preprocessing import DataPreprocessor
import argparse

def preprocess_directory(directory_path, top_n):
    preprocessed_dataframes = []
    filenames = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            try:
                processor = DataPreprocessor(file_path, top_n)
                preprocessed_dataframes.append(processor.cleaned_df)
                print(f"Processed {filename}")
                filenames.append(filename[:-4])
            except ValueError as e:
                print(f"Error processing {filename}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")
    return preprocessed_dataframes, filenames

def save_preprocessed_tfidf(preprocessed_dataframes, output_folder, filenames):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, df in enumerate(preprocessed_dataframes):
        output_filename = os.path.join(output_folder, f"preprocessed_{filenames[idx]}.csv")
        df.to_csv(output_filename, index=False)
    return

def main():
    parser = argparse.ArgumentParser(description="Preprocess a directory of CSV files using the DataPreprocessor class.")
    parser.add_argument('directory', type=str, help='Path to the directory containing the CSV files')
    parser.add_argument('top_n', type=int, help='Number of top TF-IDF words to keep')
    parser.add_argument('output_folder', type=str, help='Path to the output folder')

    args = parser.parse_args()

    preprocessed_dataframes, filenames = preprocess_directory(args.directory, args.top_n)
    save_preprocessed_tfidf(preprocessed_dataframes, args.output_folder, filenames)

if __name__ == "__main__":
    main()