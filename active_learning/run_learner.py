import os
from DataHandler import DataHandler
import argparse
from ActiveLearner import ActiveLearner
from GenerateOutput import *
import numpy as np
import time

# run a command line interface for the active learner
# example usage: python3 active_learning/run_learner.py preprocessed_data output Hall 4 0.25 10 50

def main():
    parser = argparse.ArgumentParser(description="Run the active learner on a preprocessed dataset.")
    parser.add_argument('preprocessed_data_folder', type=str, help='Path to the folder containing the preprocessed data', default='preprocessed_data')
    parser.add_argument('output_folder', type=str, help='Path to the folder to save the output files', default='output')
    parser.add_argument('filename', type=str, help='Name of the file to run the active learner on (without the .csv extension)', default='Hall')
    # parser.add_argument('initial_samples_yes', type=int, help='Number of initial samples to select from the "yes" class', default=4)
    # parser.add_argument('sample_ratio', type=float, help='Ratio of "yes" samples to "no" samples in the initial sample set', default=0.25)
    parser.add_argument('iteration_type', type=str, help=' all or few - Run active learning on all samples? or custom - specified in no_iterations', default='all')
    parser.add_argument('no_iterations', type=int, help='Number of iterations to run the active learner', default=10)
    parser.add_argument('no_statistical_validation', type=int, help='Number of iterations to run the active learner', default=20)
    parser.add_argument('tfidf', type=int, help='top tfidf feature count', default=50)

    args = parser.parse_args()

    try:
        file_path = os.path.join(args.preprocessed_data_folder, f"preprocessed_{args.filename}.csv")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return

    print(f"Running active learner on {file_path}")
    
    OUTPUT_FOLDER = args.output_folder + '/' + args.filename

    perform_stat_validation_on_multiple_start(file_path, args.filename, OUTPUT_FOLDER, args.no_statistical_validation, args.iteration_type, args.no_iterations, args.tfidf)

def perform_stat_validation_on_multiple_start(file_path, filename, output_folder, no_statistical_validation, iteration_type, no_iterations, top_tfidf):
    sample_ratio = 0.25
    initial_samples_yes = [8, 16, 32]

    fifties = []
    twenty_fives = []
    seventy_fives = []

    for no_yes in initial_samples_yes:
        sub_output_folder = output_folder + '/' + str(no_yes) + '_' + str(no_statistical_validation) + '_' + str(no_iterations)
        os.makedirs(sub_output_folder, exist_ok=True)
        # time the time to perform stat validation
        st = time.time()
        per_50, per_25, per_75, itr, baseline_recall = perform_stat_validation(file_path, filename, sub_output_folder, no_statistical_validation, iteration_type, no_iterations, no_yes, sample_ratio)
        et = time.time()
        print(f"Avg Time taken for active learner to learn over {iteration_type} iterations: {(et-st)/no_statistical_validation}")
        fifties.append(per_50)
        twenty_fives.append(per_25)
        seventy_fives.append(per_75)
        
    # put the different initial yes sample starts in the same graph
    create_combined_graph(fifties, twenty_fives, seventy_fives, baseline_recall, initial_samples_yes, filename, output_folder, itr, no_statistical_validation, top_tfidf)

def perform_stat_validation(file_path, filename, output_folder, no_statistical_validation, iteration_type, no_iterations, initial_samples_yes, sample_ratio):
    data_handler = DataHandler(file_path)
    learner = ActiveLearner(data_handler)

    output_file = open(output_folder + '/output.txt', 'a+')

    if iteration_type == 'all':
        no_iterations = int((data_handler.current_main_set.shape[0] - (initial_samples_yes + (1/sample_ratio)*initial_samples_yes))  * 0.9) 
    
    total_recall = np.zeros((no_statistical_validation, no_iterations))

    baseline_recall = generate_baseline_performance(data_handler.current_main_set['label'], output_folder, filename)

    # split the data for statistical validation - 20 splits with 90% of the data in each split
    print(f"Filename: {filename} with {initial_samples_yes} initial yes samples")
    print(f"Running statistical validation for {no_statistical_validation} runs:")
    output_file.write(f"Filename: {filename}\n")
    output_file.write(f"Running statistical validation for {no_statistical_validation} runs:\n")

    for i in range(no_statistical_validation):
        st = time.time()

        data_handler.resample_main_set(0.9)

        learner.select_initial_data(initial_samples_yes, sample_ratio)
        recalls = learner.run_active_learning(output_folder, no_iterations)

        et = time.time()
        print(f"Avg time taken by learner per iteration: {(et-st)/no_iterations}")

        total_recall[i] = recalls

        print(f"Recalls for statistical validation run {i+1}: {recalls}")
        output_file.write(f"Recalls for statistical validation run {i+1}: {recalls}\n")

        # plot the recall vs number of samples for the active learner and the baseline

        if i == no_statistical_validation - 1:
            create_active_learning_graph(recalls, baseline_recall, filename, output_folder+'/')

    print(f"Mean recall: {np.mean(total_recall, axis=0)}")
    print(f"Standard deviation of recall: {np.std(total_recall, axis=0)}")
    output_file.write(f"Mean recall: {np.mean(total_recall, axis=0)}\n")
    output_file.write(f"Standard deviation of recall: {np.std(total_recall, axis=0)}\n")

    create_stat_graph(np.percentile(total_recall, q = 50, axis=0), np.percentile(total_recall, q = 25, axis=0), np.percentile(total_recall, q=75, axis=0), baseline_recall, filename, output_folder+'/', no_iterations, no_statistical_validation, initial_samples_yes)

    return np.percentile(total_recall, q = 50, axis=0), np.percentile(total_recall, q = 25, axis=0), np.percentile(total_recall, q=75, axis=0), no_iterations, baseline_recall

if __name__ == "__main__":
    main()