import os
from DataHandler import DataHandler
import argparse
from ActiveLearner import ActiveLearner

# run a command line interface for the active learner
# example usage: python3 active_learning/run_learner.py preprocessed_data Hall 4 0.25 10

def main():
    parser = argparse.ArgumentParser(description="Run the active learner on a preprocessed dataset.")
    parser.add_argument('preprocessed_data_folder', type=str, help='Path to the folder containing the preprocessed data', default='preprocessed_data')
    parser.add_argument('filename', type=str, help='Name of the file to run the active learner on (without the .csv extension)', default='Hall')
    parser.add_argument('initial_samples_yes', type=int, help='Number of initial samples to select from the "yes" class', default=4)
    parser.add_argument('sample_ratio', type=float, help='Ratio of "yes" samples to "no" samples in the initial sample set', default=0.25)
    parser.add_argument('no_iterations', type=int, help='Number of iterations to run the active learner', default=10)
    
    args = parser.parse_args()

    try:
        file_path = os.path.join(args.preprocessed_data_folder, f"preprocessed_{args.filename}.csv")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return
    
    print(f"Running active learner on {file_path}")

    data_handler = DataHandler(file_path)
    learner = ActiveLearner(data_handler)
    
    learner.run_initial(args.initial_samples_yes, args.sample_ratio)
    learner.run_active_learning(args.no_iterations)

if __name__ == "__main__":
    main()