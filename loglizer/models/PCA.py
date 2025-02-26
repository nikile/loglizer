import numpy as np
import pandas as pd


class PCA(object):

    def __init__(self, n_components=0.95, threshold=None, c_alpha=4.4172):
        """
            The PCA model for anomaly detection

            Args:
                proj_C: The projection matrix for projecting feature vector to abnormal space
            n_components: float/int, number of principal compnents or the variance ratio they cover
            threshold: float, the anomaly detection threshold. When setting to None, the threshold
                is automatically caculated using Q-statistics
            c_alpha: float, the c_alpha parameter for caculating anomaly detection threshold using
                Q-statistics. The following is lookup table for c_alpha:
                c_alpha = 1.7507; # alpha = 0.08
                c_alpha = 1.9600; # alpha = 0.05
                c_alpha = 2.5758; # alpha = 0.01
                c_alpha = 2.807; # alpha = 0.005
                c_alpha = 2.9677;  # alpha = 0.003
                c_alpha = 3.2905;  # alpha = 0.001
                c_alpha = 3.4808;  # alpha = 0.0005
                c_alpha = 3.8906;  # alpha = 0.0001
                c_alpha = 4.4172;  # alpha = 0.00001

        """
        self.proj_C = None
        self.components = None
        self.n_components = n_components
        self.threshold = threshold
        self.c_alpha = c_alpha


    def fit(self, X):
        """
            Fits PCA model on train data

            Args:
                X: ndarray, the event count matrix of shape num_instances-by-num_events

        """
        print('====== Model summary ======')
        num_instances, num_events = X.shape
        X_cov = np.dot(X.T, X) / float(num_instances)
        U, sigma, V = np.linalg.svd(X_cov)
        n_components = self.n_components
        if n_components < 1:
            total_variance = np.sum(sigma)
            variance = 0
            for i in range(num_events):
                variance += sigma[i]
                if variance / total_variance >= n_components:
                    break
            n_components = i + 1

        P = U[:, :n_components]
        I = np.identity(num_events, int)
        self.components = P
        self.proj_C = I - np.dot(P, P.T)
        print('n_components: {}'.format(n_components))
        print('Project matrix shape: {}-by-{}'.format(self.proj_C.shape[0], self.proj_C.shape[1]))

        if not self.threshold:
            # Calculate threshold using Q-statistic. Information can be found at:
            # http://conferences.sigcomm.org/sigcomm/2004/papers/p405-lakhina111.pdf
            phi = np.zeros(3)
            for i in range(3):
                for j in range(n_components, num_events):
                    phi[i] += np.power(sigma[j], i + 1)
            h0 = 1.0 - 2 * phi[0] * phi[2] / (3.0 * phi[1] * phi[1])
            self.threshold = phi[0] * np.power(self.c_alpha * np.sqrt(2 * phi[1] * h0 * h0) / phi[0]
                                               + 1.0 + phi[1] * h0 * (h0 - 1) / (phi[0] * phi[0]),
                                               1.0 / h0)
        print('SPE threshold: {}\n'.format(self.threshold))


    def get_SPE_for_single_seq(self, single_seq):
        y_a = np.dot(self.proj_C, single_seq)
        SPE = np.dot(y_a, y_a)
        return SPE


    def predict(self, X):
        """
            Predict anomalies on given data

            Args:
                X: array of lists of event id's np.array(['E21', 'E22', ...], [...],...)
            Returns:
                y_pred: array of predictions np.array(1., 0.,...)
                        1. - anomaly

        """
        assert self.proj_C is not None, 'PCA model needs to be trained before prediction.'
        y_pred = [0] * X.shape[0]
        SPE_pred = [0] * X.shape[0]
        for i in range(X.shape[0]):
            SPE = self.get_SPE_for_single_seq(X[i, :])
            SPE_pred[i] = SPE
            if SPE > self.threshold:
                y_pred[i] = 1
        return pd.DataFrame({'SPE':SPE_pred,'prediction' :y_pred})

