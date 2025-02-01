import sklearn as sk
import numpy as np

class ActiveLearner:
    def __init__(self, data_handler):
        self.data_handler = data_handler

    def select_initial_data(self, no_samples_yes, sample_ratio=0.25):
        self.data_handler.select_samples_initial(no_samples_yes, sample_ratio)
    
    def weighted_best_from_rest(self, probabilities, q):
        """
        Compute the acquisition function (Best + q * Rest) / |q * Best - Rest| for binary classification.
        """
        # Extract Best and Rest probabili
        # ties
        best = np.max(probabilities, axis=1)  # Best confidence score
        rest = np.min(probabilities, axis=1)  # Rest confidence score

        # Compute the acquisition function
        scores = (best + q * rest) / (np.abs(q * best - rest) + 1e-10)
        return scores
    
    def run_active_learning(self, output_folder, no_iterations, initial_q = 1):

        final_q = 0
        labeling_budget = 100

        recalls = np.zeros(no_iterations)

        for i in range(no_iterations):
            if i < labeling_budget:
                q = initial_q - (initial_q - final_q) * i / no_iterations
            else:
                q = final_q

            # print(f"############# Running iteration {i+1}")

            clf = sk.naive_bayes.GaussianNB()
            clf.fit(self.data_handler.current_X, np.squeeze(self.data_handler.current_y))

            # evaluate the classifier on data_handler.current_main_set and select the next sample

            current_main_set_X = self.data_handler.current_main_set.drop('label', axis=1)
            current_main_set_y = self.data_handler.current_main_set['label']

            # calculate precision recall and F1 score

            # get predictions on the entire dataset
            predictions = clf.predict(self.data_handler.test_X)

            # get probabilities on the remaining dataset to acquire new samples
            probabilities = clf.predict_proba(current_main_set_X)

            # Performance on entire dataset
            accuracy = sk.metrics.accuracy_score(self.data_handler.test_y, predictions)
            precision = sk.metrics.precision_score(self.data_handler.test_y, predictions, average='binary', pos_label=1)
            recall = sk.metrics.recall_score(self.data_handler.test_y, predictions, average='binary', pos_label=1)
            f1 = sk.metrics.f1_score(self.data_handler.test_y, predictions, average='binary', pos_label=1)
            
            # print(f"Iteration {i+1}, Accuracy: {accuracy}")
            # print(f"Iteration {i+1}, Precision: {precision}")
            # print(f"Iteration {i+1}, Recall: {recall}")
            # print(f"Iteration {i+1}, F1: {f1}")

            # calculate the acquisition function`s scores
            scores = self.weighted_best_from_rest(probabilities, q)

            self.data_handler.select_next_active_learning_sample(scores, 1)

            recalls[i] = recall
        return recalls