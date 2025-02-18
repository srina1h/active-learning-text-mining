import sklearn as sk
import numpy as np
from typing import Tuple, Union

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
    
    def ucb_acquisition_proba(self, predict_proba, kappa=1.0):
        """
        Compute a UCB-like acquisition score using predicted probabilities.

        Parameters:
        - predict_proba: A 2D numpy array of shape (n_samples, 2) 
                        with class probabilities (assume column 1 is pos-class probability).
        - kappa: A constant to trade off exploitation and exploration.

        Returns:
        - acquisition_values: An array of UCB-like scores for each sample.
        """
        # Extract the probability for the positive class
        p = predict_proba[:, 1]
        # Compute uncertainty as the standard deviation of a Bernoulli distribution
        uncertainty = np.sqrt(p * (1 - p))
        # Combine the probability and uncertainty in a UCB-style rule
        acquisition_values = p + kappa * uncertainty
        return acquisition_values

    def run_active_learning(self, output_folder, no_iterations, model_type, initial_q = 1):
        final_q = 0
        labeling_budget = 100
        recalls = np.zeros(no_iterations)

        match model_type:
            case 'NB':
                clf = sk.naive_bayes.GaussianNB()
                clf.fit(self.data_handler.current_X, np.squeeze(self.data_handler.current_y))

                recalls[0] = sk.metrics.recall_score(self.data_handler.test_y, clf.predict(self.data_handler.test_X), average='binary', pos_label=1)

                predictions = clf.predict(self.data_handler.test_X)

                for i in range(1, no_iterations):
                    if i < labeling_budget:
                        q = initial_q - (initial_q - final_q) * i / no_iterations
                    else:
                        q = final_q

                    current_main_set_X = self.data_handler.current_main_set.drop('label', axis=1)

                    probabilities = clf.predict_proba(current_main_set_X)

                    scores = self.weighted_best_from_rest(probabilities, q)

                    sample_x, sample_y = self.data_handler.select_next_active_learning_sample(scores)

                    clf.partial_fit(sample_x, sample_y)

                    new_preds = clf.predict(self.data_handler.test_X)

                    # Performance on entire dataset
                    recall = sk.metrics.recall_score(self.data_handler.test_y, new_preds, average='binary', pos_label=1)
                    recalls[i] = recall
            case 'GPM':
                # Gaussian Process model
                kernel = 1.0 * sk.gaussian_process.kernels.RBF(length_scale=1.0)
                # gpc = GPCWithVariance(kernel=kernel)
                gpc = sk.gaussian_process.GaussianProcessClassifier(kernel=kernel, n_jobs=-1)
                
                gpc.fit(self.data_handler.current_X, np.squeeze(self.data_handler.current_y))

                # Initial evaluation
                recalls[0] = sk.metrics.recall_score(self.data_handler.test_y, gpc.predict(self.data_handler.test_X), average='binary', pos_label=1)

                # Active learning loop
                skipped_itrs = 0
                prev_recall = recalls[0]
                for i in range(1, no_iterations):
                    print(f"Iteration {i}")
                    if i % 1000 == 0 or i == no_iterations - 1:
                        current_main_set_X = self.data_handler.current_main_set.drop('label', axis=1)

                        scores = self.ucb_acquisition_proba(gpc.predict_proba(current_main_set_X), 1)

                        _, _ = self.data_handler.select_next_active_learning_sample(scores, skipped_itrs)
                        skipped_itrs = 0

                        gpc.fit(self.data_handler.current_X, np.squeeze(self.data_handler.current_y))

                        # evaluate the classifier on entire dataset
                        predictions = gpc.predict(self.data_handler.test_X)

                        # Performance on entire dataset
                        prev_recall = sk.metrics.recall_score(self.data_handler.test_y, predictions, average='binary', pos_label=1)
                    else:
                        skipped_itrs += 1
                    recalls[i] = prev_recall
        return recalls