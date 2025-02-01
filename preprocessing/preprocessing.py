"""
Example usage: python3 preprocessing/preprocess_folder.py data_folder/ 50 preprocessed_data/
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

class DataPreprocessor:
    def __init__(self, file_path, top_n):
        """
        Initialize the DataPreprocessor with the path to the CSV file.
        Load the data, preprocess the 'Abstract' column, and store the result.
        """
        self.file_path = file_path
        self.data = self.load_data()
        self.data['Processed_Abstract'] = self.preprocess_abstract(self.data['Abstract'])
        self.data, self.top_tfidf_words, self.top_n_tfidf = self.calculate_top_tfidf(self.data, top_n)
        self.tfidf_frequency_df = self.calculate_frequency_of_top_tfidf_words(self.data)
        self.data = pd.concat([self.data, self.tfidf_frequency_df], axis=1)

        #create a new cleaned df with only tfidf frequencies and label
        self.cleaned_df = self.data.drop(columns=['Abstract', 'Processed_Abstract', 'Year', 'Document Title', 'PDF Link'])
        
        #check if these 3 columns exist and drop them if they do - 'Authors', 'Source', 'abs'
        if 'Authors' in self.cleaned_df.columns:
            self.cleaned_df = self.cleaned_df.drop(columns=['Authors'])
        if 'Source' in self.cleaned_df.columns:
            self.cleaned_df = self.cleaned_df.drop(columns=['Source'])
        if 'abs' in self.cleaned_df.columns:
            self.cleaned_df = self.cleaned_df.drop(columns=['abs'])

        # change the label to binary from yes no to 0, 1
        self.cleaned_df['label'] = self.cleaned_df['label'].apply(lambda x: 1 if x == 'yes' else 0)

    def load_data(self):
        """
        Load the CSV file into a pandas DataFrame.
        """
        encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
        try:
            for encoding in encodings:
                try:
                    data = pd.read_csv(self.file_path, encoding=encoding)
                    print(f"Successfully read with encoding: {encoding}")
                    return data
                except UnicodeDecodeError:
                    print(f"Failed to decode with encoding: {encoding}")
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

    def preprocess_abstract(self, abstracts):
        """
        Preprocess the 'Abstract' column by applying tokenization, stop word removal, and stemming.
        """
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()

        processed_abstracts = []
        for abstract in abstracts:
            if pd.isna(abstract):
                processed_abstracts.append('')
                continue

            # Tokenization
            tokens = word_tokenize(abstract)

            # Stop word removal
            filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

            # Stemming
            stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

            # Join tokens back into a single string
            processed_abstracts.append(' '.join(stemmed_tokens))

        return processed_abstracts

    def calculate_top_tfidf(self, data, top_n):
        """
        Calculate TF-IDF values for the processed abstracts and retain only the top 100 words.
        """
        print(data.shape)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data['Processed_Abstract'])


        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        # Sum TF-IDF values for each word across all documents
        word_scores = tfidf_df.sum(axis=0).sort_values(ascending=False)

        # Get the top 100 words by TF-IDF score
        top_100_words = word_scores.head(top_n).index.tolist()

        # Filter the TF-IDF DataFrame to include only the top 100 words
        filtered_tfidf_df = tfidf_df[top_100_words]

        # print("Filtered TF-IDF DataFrame:")
        # print(filtered_tfidf_df.head())

        return data, top_100_words, filtered_tfidf_df
    
    def calculate_frequency_of_top_tfidf_words(self, data):
        """
        Calculate the frequency of each top TF-IDF word within each abstract
        """
        tftdf_frequency_df = pd.DataFrame(columns=self.top_tfidf_words)
        for index, row in data.iterrows():
            abstract_words = row['Processed_Abstract'].split()
            word_counts = {word: abstract_words.count(word) for word in self.top_tfidf_words}
            tftdf_frequency_df.loc[index] = word_counts
        return tftdf_frequency_df