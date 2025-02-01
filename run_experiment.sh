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
OUTPUT_DIR=test_op
PREPROCESSED_DATA_DIR=preprocessed_data

# Runtime parameters
iteration_type=few
no_iterations=100
no_statistical_validation=5

# create the output directory if it does not exist
mkdir -p $OUTPUT_DIR

for file in Hall Kitchenham Wahono Radjenovic
do
    mkdir -p $OUTPUT_DIR/$file
    python3 $FILENAME $PREPROCESSED_DATA_DIR $OUTPUT_DIR $file $iteration_type $no_iterations $no_statistical_validation
done