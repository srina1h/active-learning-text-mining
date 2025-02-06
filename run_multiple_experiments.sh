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
RAW_DATA_DIR=data_folder
# OUTPUT_DIR=test_op
# PREPROCESSED_DATA_DIR=preprocessed_data

# Runtime parameters
iteration_type=all
no_iterations=100
no_statistical_validation=30

for top_tf_idf in 50 25 10 5
do
    OUTPUT_DIR=test_op_$top_tf_idf
    PREPROCESSED_DATA_DIR=preprocessed_data_$top_tf_idf
    mkdir -p $OUTPUT_DIR
    mkdir -p $PREPROCESSED_DATA_DIR

    python3 preprocessing/preprocess_folder.py $RAW_DATA_DIR/ $top_tf_idf $PREPROCESSED_DATA_DIR/

    for file in Hall Kitchenham Wahono Radjenovic
    do
        mkdir -p $OUTPUT_DIR/$file
        python3 $FILENAME $PREPROCESSED_DATA_DIR $OUTPUT_DIR $file $iteration_type $no_iterations $no_statistical_validation $top_tf_idf
    done
done