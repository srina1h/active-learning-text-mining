# Script to run multiple experiments with different parameters on run_learner.py

# Parameters to vary:
# 1. initial_samples_yes
# 2. sample_ratio
# 3. no_iterations
# 4. no_statistical_validation

# Usage: bash run_experiment.sh

# Experiment 1: Vary initial_samples_yes

# set global output directory
FILENAME=active_learning/run_learner.py
OUTPUT_DIR=output
PREPROCESSED_DATA_DIR=preprocessed_data

# Runtime parameters
no_iterations=20
no_statistical_validation=100
inital_split_ratio=0.25

# create the output directory if it does not exist
mkdir -p $OUTPUT_DIR

for file in Hall Kitchenham Wahono Radjenovic
do
    for initial_samples_yes in 4 8 16 32 64
    do
        # run the learner and store the output in the output directory specific to the experiment
        mkdir -p $OUTPUT_DIR/$file/$initial_samples_yes\_$inital_split_ratio\_$no_iterations\_$no_statistical_validation
        echo $FILENAME $PREPROCESSED_DATA_DIR $OUTPUT_DIR/$file/$initial_samples_yes\_$inital_split_ratio\_$no_iterations\_$no_statistical_validation $file $initial_samples_yes $inital_split_ratio $no_iterations $no_statistical_validation
        python3 $FILENAME $PREPROCESSED_DATA_DIR $OUTPUT_DIR/$file/$initial_samples_yes\_$inital_split_ratio\_$no_iterations\_$no_statistical_validation $file $initial_samples_yes $inital_split_ratio $no_iterations $no_statistical_validation > $OUTPUT_DIR/$file/$initial_samples_yes\_$inital_split_ratio\_$no_iterations\_$no_statistical_validation/output.txt
    done
done