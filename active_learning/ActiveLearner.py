import sklearn as sk
import numpy as np
from modAL.models import ActiveLearner as AL
from modAL.uncertainty import uncertainty_sampling

class ActiveLearner:
    def __init__(self, data_handler):
        self.data_handler = data_handler

    def select_initial_data(self, no_samples_yes, sample_ratio=0.25):
        self.data_handler.select_samples_initial(no_samples_yes, sample_ratio)
    
    def weighted_best_from_rest(self, probabilities, q):
        """
        Compute the acquisition function (Best + q * Rest) / |q * Best - Rest| for binary classification.
        """
        # Extract Best and Rest probabilities
        best = np.max(probabilities, axis=1)  # Best confidence score
        rest = np.min(probabilities, axis=1)  # Rest confidence score

        # Compute the acquisition function
        scores = (best + q * rest) / (np.abs(q * best - rest) + 1e-10)
        return scores

    def GP_regression_std(regressor, X):
        _, std = regressor.predict(X, return_std=True)
        query_idx = np.argmax(std)
        return query_idx, X[query_idx]
    
    def run_active_learning(self, output_folder, no_iterations, model_type, initial_q = 1,):
        final_q = 0
        labeling_budget = 100
        recalls = np.zeros(no_iterations)

        match model_type:
                case 'NB':
                    for i in range(no_iterations):
                            if i < labeling_budget:
                                q = initial_q - (initial_q - final_q) * i / no_iterations
                            else:
                                q = final_q

                            clf = sk.naive_bayes.GaussianNB()
                            clf.fit(self.data_handler.current_X, np.squeeze(self.data_handler.current_y))

                            # evaluate the classifier on data_handler.current_main_set and select the next sample

                            current_main_set_X = self.data_handler.current_main_set.drop('label', axis=1)

                            # get predictions on the entire dataset
                            predictions = clf.predict(self.data_handler.test_X)

                            # get probabilities on the remaining dataset to acquire new samples
                            probabilities = clf.predict_proba(current_main_set_X)

                            # Performance on entire dataset
                            recall = sk.metrics.recall_score(self.data_handler.test_y, predictions, average='binary', pos_label=1)

                            # calculate the acquisition function`s scores
                            scores = self.weighted_best_from_rest(probabilities, q)

                            self.data_handler.select_next_active_learning_sample(scores, 1)

                            recalls[i] = recall
                case 'GPM':
                    # Gaussian Process model
                    kernel = sk.gaussian_process.kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + sk.gaussian_process.kernels.WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
                    regressor = sk.gaussian_process.GaussianProcessRegressor(kernel=kernel)

                    # Active learner
                    learner = AL(
                        estimator=regressor,
                        query_strategy=GP_regression_std,  # Define your query strategy
                        X_training=self.data_handler.current_X, 
                        y_training=self.data_handler.current_y
                    )

                    # Initial evaluation
                    recalls[0] = sk.metrics.recall_score(self.data_handler.test_y, learner.predict(self.data_handler.test_X), average='binary', pos_label=1)

                    # Active learning loop
                    for i in range(1, no_iterations):
                        current_main_set_X = self.data_handler.current_main_set.drop('label', axis=1)

                        query_idx, _ = learner.query(current_main_set_X)
                        sample_x, sample_y = self.data_handler.select_next_with_idx(query_idx)
                        learner.teach(sample_x, sample_y)

                        # evaluate the classifier on entire dataset
                        predictions = learner.predict(self.data_handler.test_X)

                        # Performance on entire dataset
                        recall = sk.metrics.recall_score(self.data_handler.test_y, predictions, average='binary', pos_label=1)
                        recalls[i] = recall
        return recalls