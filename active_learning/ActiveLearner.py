import sklearn as sk
import numpy as np
from typing import Tuple, Union

class GPCWithVariance(sk.gaussian_process.GaussianProcessClassifier):

    def __init__(self, kernel = 'rbf', n_jobs = -1):
        self.kernel = kernel
        self.n_jobs = n_jobs
        super().__init__(kernel=kernel, n_jobs=n_jobs)

    def predict_a(
        self, X: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Return estimates of the latent function for X.
        
        Notes:
        ------
        - For binary classification (n_classes = 2), the output shape is
        (n_samples,).
        - For multi-class classification, the output shape is (n_samples,
        n_classes) when multi_class="one_vs_rest", and is shaped (n_samples,
        n_classes*(n_classes - 1)/2) when multi_class="one_vs_one". In other
        terms, There are as many columns as trained Binary GPC sub-models.
        - The number of classes (n_classes) is determined by the number of
        unique target values in the training data.
        """
        sk.utils.validation.check_is_fitted(self)

        if self.n_classes_ > 2:  # Multi-class case
            f_stars = []
            std_f_stars = []
            for estimator, kernel in zip(self.base_estimator_.estimators_, self.kernel_.kernels):
                result = self._binary_predict_a(estimator, kernel, X, return_std)
                if not return_std:
                    f_stars.append(result)
                else:
                    f_stars.append(result[0])
                    std_f_stars.append(result[1])

            if not return_std:
                return np.array(f_stars).T

            return np.array(f_stars).T, np.array(std_f_stars).T
        else:  # Binary case
            return self._binary_predict_a(self.base_estimator_, self.kernel_, X, return_std)

    @staticmethod
    def _binary_predict_a(estimator, kernel, X, return_std):
        """ Return mean and std of the latent function estimates for X. """
        sk.utils.validation.check_is_fitted(estimator)

        # Based on Algorithm 3.2 of GPML
        K_star = kernel(estimator.X_train_, X)  # K_star = k(x_star)
        f_star = K_star.T.dot(estimator.y_train_ - estimator.pi_)  # Line 4
        if not return_std:
            return f_star

        v = np.linalg.solve(estimator.L_, estimator.W_sr_[:, np.newaxis] * K_star)  # Line 5
        # Line 6 (compute np.diag(v.T.dot(v)) via einsum)
        var_f_star = kernel.diag(X) - np.einsum("ij,ij->j", v, v)

        return f_star, np.sqrt(var_f_star)

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

    def GP_clf_Var(self, gpc, X):
        _, latent_variance = gpc.predict_a(X, return_std=True)
        most_uncertain_index = np.argsort(latent_variance)[-1:]
        return most_uncertain_index
    
    def GP_entropy(self, gpc, X):
        preds = gpc.predict_proba(X)
        entropy = -np.sum(preds * np.log(preds), axis=1)
        most_uncertain_index = np.argsort(entropy)[-1:]
        return most_uncertain_index

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

                extra_labeling = 50

                # Initial evaluation
                recalls[0] = sk.metrics.recall_score(self.data_handler.test_y, gpc.predict(self.data_handler.test_X), average='binary', pos_label=1)

                # Active learning loop
                for i in range(1, no_iterations):
                    print(f"Iteration {i}")
                    current_main_set_X = self.data_handler.current_main_set.drop('label', axis=1)

                    if extra_labeling > 0:
                        # query_idx = self.GP_clf_Var(gpc, current_main_set_X)
                        # sample random 100 samples from the dataset
                        random_100_set = np.random.choice(current_main_set_X.index, size=100, replace=False)
                        # now calculate the entropy of the random 100 samples
                        query_idx = self.GP_entropy(gpc, current_main_set_X.loc[random_100_set])
                        # find a sample in the current_main_set using the query_idx

                        _, _ = self.data_handler.select_next_with_idx(query_idx)

                        gpc.fit(self.data_handler.current_X, np.squeeze(self.data_handler.current_y))
                        extra_labeling -= 1

                    # evaluate the classifier on entire dataset
                    predictions = gpc.predict(self.data_handler.test_X)

                    # Performance on entire dataset
                    recall = sk.metrics.recall_score(self.data_handler.test_y, predictions, average='binary', pos_label=1)
                    recalls[i] = recall
        return recalls