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
        scores = (best + q * rest) / np.abs(q * best - rest)
        return scores
    
    def run_active_learning(self, no_iterations, initial_q = 1):
        # run a scikitlearn naive bayes classifier

        final_q = 0 

        recalls = np.zeros(no_iterations)

        for i in range(no_iterations):
            q = initial_q - (initial_q - final_q) * i / no_iterations

            print(f"############# Running iteration {i+1}")
            clf = sk.naive_bayes.GaussianNB()
            clf.fit(self.data_handler.current_X, np.squeeze(self.data_handler.current_y))

            # evaluate the classifier on data_handler.current_main_set and select the next sample

            current_main_set_X = self.data_handler.current_main_set.drop('label', axis=1)
            current_main_set_y = self.data_handler.current_main_set['label']

            # calculate precision recall and F1 score

            predictions = clf.predict(current_main_set_X)
            probabilities = clf.predict_proba(current_main_set_X)

            accuracy = sk.metrics.accuracy_score(current_main_set_y, predictions)
            precision = sk.metrics.precision_score(current_main_set_y, predictions, average='binary', pos_label='yes')
            recall = sk.metrics.recall_score(current_main_set_y, predictions, average='binary', pos_label='yes')
            f1 = sk.metrics.f1_score(current_main_set_y, predictions, average='binary', pos_label='yes')
            
            print(f"Iteration {i+1}, Accuracy: {accuracy}")
            print(f"Iteration {i+1}, Precision: {precision}")
            print(f"Iteration {i+1}, Recall: {recall}")
            print(f"Iteration {i+1}, F1: {f1}")

            # calculate the acquisition function`s scores
            scores = self.weighted_best_from_rest(probabilities, q)

            self.data_handler.select_next_active_learning_sample(scores, 1)

            # decrease q by 0.1 to favor exploitation over exploration
            q -= 0.1

            recalls[i] = recall
        return recalls