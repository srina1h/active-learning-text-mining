import os
from DataHandler import DataHandler
import argparse
from ActiveLearner import ActiveLearner
from GenerateOutput import *
import numpy as np

# run a command line interface for the active learner
# example usage: python3 active_learning/run_learner.py preprocessed_data output Hall 4 0.25 10 50

def main():
    parser = argparse.ArgumentParser(description="Run the active learner on a preprocessed dataset.")
    parser.add_argument('preprocessed_data_folder', type=str, help='Path to the folder containing the preprocessed data', default='preprocessed_data')
    parser.add_argument('output_folder', type=str, help='Path to the folder to save the output files', default='output')
    parser.add_argument('filename', type=str, help='Name of the file to run the active learner on (without the .csv extension)', default='Hall')
    parser.add_argument('initial_samples_yes', type=int, help='Number of initial samples to select from the "yes" class', default=4)
    parser.add_argument('sample_ratio', type=float, help='Ratio of "yes" samples to "no" samples in the initial sample set', default=0.25)
    parser.add_argument('no_iterations', type=int, help='Number of iterations to run the active learner', default=10)
    parser.add_argument('no_statistical_validation', type=int, help='Number of iterations to run the active learner', default=20)

    args = parser.parse_args()

    try:
        file_path = os.path.join(args.preprocessed_data_folder, f"preprocessed_{args.filename}.csv")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return

    print(f"Running active learner on {file_path}")

    data_handler = DataHandler(file_path)
    learner = ActiveLearner(data_handler)

    baseline_recall = generate_baseline_performance(data_handler.current_main_set['label'], args.output_folder+'/', args.filename)

    # split the data for statistical validation - 20 splits with 90% of the data in each split

    total_recall = np.zeros((args.no_statistical_validation, args.no_iterations))

    for i in range(args.no_statistical_validation):

        data_handler.resample_main_set(0.9)

        learner.select_initial_data(args.initial_samples_yes, args.sample_ratio)
        recalls = learner.run_active_learning(args.no_iterations)

        total_recall[i] = recalls

        # plot the recall vs number of samples for the active learner and the baseline

        if i == args.no_statistical_validation - 1:
            create_active_learning_graph(recalls, baseline_recall, args.filename, args.output_folder+'/')

    create_stat_graph(np.mean(total_recall, axis=0), np.std(total_recall, axis=0), args.filename, args.output_folder+'/', args.no_iterations, args.no_statistical_validation)
    
if __name__ == "__main__":
    main()