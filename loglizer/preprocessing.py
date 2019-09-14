"""
The interface for data preprocessing.

Authors:
    LogPAI Team

"""

import numpy as np
import pandas as pd
from collections import Counter

class FeatureExtractor(object):


    def __init__(self):
        self.idf_vec = None
        self.mean_vec = None
        self.events = None


    def df_fit_transform(self, X_seq):
        """
            Fit and transform the data matrix.
            Variant of a similar to "fit_transform" function,
            but with using a pandas lib for more convenient debugging.

            Args:
                X_seq: ndarray, log sequences matrix

            Returns:
                X_new: The transformed data matrix
        """
        print('====== Transformed train data summary ======')

        x_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            x_counts.append(event_counts)
        X_df = pd.DataFrame(x_counts)
        X_df = X_df.fillna(0)
        self.events = X_df.columns

        num_instance, num_event = X_df.shape

        #tf - idf term weighting
        df_vec = np.sum(X_df > 0, axis=0)
        self.idf_vec = np.log(num_instance / (df_vec + 1e-8))

        idf_matrix = X_df * np.tile(self.idf_vec, (num_instance, 1))
        X = idf_matrix

        #zero-mean normalization
        mean_vec = X.mean(axis=0)
        self.mean_vec = mean_vec.values.reshape(1, num_event)
        X = X - np.tile(self.mean_vec, (num_instance, 1))

        X_new = X

        print('Train data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1]))
        return X_new


    def fit_transform(self, X_seq):
        """
            Fit and transform the data matrix

            Args:
                X_seq: ndarray, log sequences matrix

            Returns:
                X_new: The transformed data matrix
        """
        print('====== Transformed train data summary ======')

        x_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            x_counts.append(event_counts)
        X_df = pd.DataFrame(x_counts)
        X_df = X_df.fillna(0)
        self.events = X_df.columns
        X = X_df.values

        num_instance, num_event = X.shape

        #tf - idf term weighting
        df_vec = np.sum(X > 0, axis=0)
        self.idf_vec = np.log(num_instance / (df_vec + 1e-8))
        idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
        X = idf_matrix

        #zero-mean normalization
        mean_vec = X.mean(axis=0)
        self.mean_vec = mean_vec.reshape(1, num_event)
        X = X - np.tile(self.mean_vec, (num_instance, 1))

        X_new = X

        print('Train data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1]))
        return X_new


    def transform(self, X_seq):
        """
            Transform the data matrix with trained parameters

            Args:
                X_seq: log sequences matrix

            Returns:
                X_new: The transformed data matrix
        """
        print('====== Transformed test data summary ======')
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        empty_events = set(self.events) - set(X_df.columns)
        for event in empty_events:
            X_df[event] = [0] * len(X_df)
        # only those events (keys) that were in the training data set are taken into account
        X = X_df[self.events].values

        num_instance, num_event = X.shape
        # tf - idf term weighting
        idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
        X = idf_matrix
        # zero-mean normalization
        X = X - np.tile(self.mean_vec, (num_instance, 1))

        X_new = X

        print('Test data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1]))

        return X_new
