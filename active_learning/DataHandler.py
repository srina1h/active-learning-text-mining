import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

class DataHandler:
    def __init__(self, file_path):
        self.full_dataset = self.load_preprocessed_data(file_path)
        self.normalize_data()
        self.resample_main_set()
        self.set_test_set()
    
    def set_test_set(self):
        self.test_X = self.full_dataset.drop('label', axis=1)
        self.test_y = self.full_dataset['label']

    def load_preprocessed_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            return None
    
    def normalize_data(self):
        # normalize all columns except the label column using the min-max scaler in sklean
        scaler = MinMaxScaler()

        self.full_dataset.iloc[:, 1:] = scaler.fit_transform(self.full_dataset.iloc[:, 1:])

    def select_samples_initial(self, no_samples_yes, sample_ratio=0.25):
        no_samples_no = int(no_samples_yes * (1/sample_ratio))

        # select 4 'yes' rows from the end and drop from main set
        yes_rows = self.current_main_set[self.current_main_set['label'] == 1].tail(no_samples_yes)
        self.current_main_set.drop(yes_rows.index, inplace=True)

        # # select 16 'no' rows at random ansd drop from main set
        # no_rows = self.current_main_set[self.current_main_set['label'] == 0].sample(no_samples_no)
        # self.current_main_set.drop(no_rows.index, inplace=True)

        # ALTERNATE METHOD FOR SELECTING NO SAMPLES (2/5/25)

        # select no_samples_no rows at random. Assume they are negative samples
        no_rows = self.current_main_set.sample(no_samples_no, random_state=42)
        self.current_main_set.drop(no_rows.index, inplace=True)
        #change the label of the selected rows to 0
        no_rows['label'] = 0

        # combine the selected rows to create the initial training set
        self.current_X = pd.concat([self.current_X, yes_rows.drop('label', axis=1), no_rows.drop('label', axis=1)])
        self.current_y = pd.concat([self.current_y, yes_rows['label'], no_rows['label']])
        return

    def select_next_active_learning_sample(self, scores, no_samples=1):
        # select thee highest scores from scores and use them to select the next samples
        # then drop these samples from the main set and add them to the current set

        next_samples = self.current_main_set.iloc[scores.argsort()[-no_samples:]]
        self.current_main_set.drop(next_samples.index, inplace=True)
        sample_x = next_samples.drop('label', axis=1)
        sample_y = next_samples['label']
        self.current_X = pd.concat([self.current_X, next_samples.drop('label', axis=1)])
        self.current_y = pd.concat([self.current_y, next_samples['label']])

        return sample_x, sample_y
    
    def select_next_with_idx(self, idx):

        next_samples = self.current_main_set.iloc[idx]
        self.current_main_set.drop(next_samples.index, inplace=True)
        sample_x = next_samples.drop('label', axis=1)
        sample_y = next_samples['label']
        self.current_X = pd.concat([self.current_X, next_samples.drop('label', axis=1)])
        self.current_y = pd.concat([self.current_y, next_samples['label']])

        return sample_x, sample_y
    
    def resample_main_set(self, sample_percent = 0.9):
        # randomly sample 90% of the data for the main set
        self.current_main_set = self.full_dataset.sample(frac=sample_percent, random_state=42)
        self.current_X = pd.DataFrame()
        self.current_y = pd.DataFrame()