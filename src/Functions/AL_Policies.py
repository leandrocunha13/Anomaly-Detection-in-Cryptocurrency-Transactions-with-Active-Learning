import numpy as np
import pandas as pd

########   UNSUPERVISED ACTIVE LEARNING POLICIES     ###########################
class AnomalyDetectionQuery:
    
    def __init__(self, model):
        self.model = model
        
    def predict_anomaly_scores(self, X_train, X_test, predict_on = 'train'):
        
        self.model.fit(X_train.values)

        #computes the anomaly scores
        if(predict_on == 'test'):
            predicted_scores = self.model.decision_function(X_test.values)
        else:
            predicted_scores = self.model.decision_scores_

        return predicted_scores
    
    def query(self, X_train, X_test, batch_size, compute_on):
        
        #Define random_state
        if(isinstance(self.model, IForest)):
            params = {"random_state": np.random.randint(0, 10000)}
            self.model.set_params(**params)
            
        scores = self.predict_anomaly_scores(X_train, X_test, compute_on)
        
        scores = np.array(scores)
        highest_idx = np.argsort(scores)[::-1][:batch_size]
        
        return highest_idx

    
class EllipticEnvelopeQuery:
    
    def __init__(self, model):
        self.model = model
        
    def validate_input_data(self, data):
    
        # Calculate the correlation matrix
        corr_matrix = data.corr()

        #Find the columns with a correlation coefficient greater than 0.95 or less than -0.95
        high_corr_cols = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= 0.9999999:
                    high_corr_cols.add(corr_matrix.columns[j])

        # Drop the high correlation columns from the DataFrame
        data = data.drop(columns=high_corr_cols)

        return data
    
    def mahalanobis_score(self, X_train, X_test, compute_on = 'test'):
        
        X_train = self.validate_input_data(X_train)

        ellenv = self.model.fit(X_train)

        if(compute_on == 'test' and X_test is not None):
            distances = ellenv.mahalanobis(X_test)
        elif(compute_on == 'train'):
            distances = ellenv.dist_

        return distances
    
    
    def query(self, X_train, X_test, batch_size, compute_on):
        
        params = {"random_state": np.random.randint(0, 10000)}
        self.model.set_params(**params)
        
        scores = self.mahalanobis_score(X_train, X_test, compute_on)
        
        scores = np.array(scores)
        highest_idx = np.argsort(scores)[::-1][:batch_size]
        
        return highest_idx
    
class RandomSampling:
    
    def query(self, X_train, X_test, batch_size, compute_on):
        
        #initial random sampling
        query_samples = X_train.sample(n=batch_size)
        selected_idx = query_samples.index
        
        return selected_idx



##########   SUPERVISED ACTIVE LEARNING POLICIES ###########

#hot learners
class UncertaintySampling:
    
    def __init__(self, model):
        self.model = model
    
    def measure_uncertainty(self, X_train, y_train, X_test):

        #Define params for a balanced prediction
        if(isinstance(self.model, XGBClassifier)):
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            ratio = float(n_neg) / n_pos
            params = {"scale_pos_weight": ratio, "random_state": np.random.randint(0, 10000)}
        else:
            params = {"class_weight": 'balanced', "random_state": np.random.randint(0, 10000)}

        #set parameters to ensure a balanced prediction
        self.model.set_params(**params)

        #Train the supervised classifier on the labeled pool
        self.model.fit(X_train, y_train)

        #get the predicted scores for the unlabeled pool
        scores= self.model.predict_proba(X_test)

        entropies = entropy(scores.T)

        return entropies
    
    def query(self, X_train, y_train, X_test, batch_size):
        
        scores = self.measure_uncertainty(X_train, y_train, X_test)
        
        scores = np.array(scores)
        highest_idx = np.argsort(scores)[::-1][:batch_size]

        return highest_idx

class ExpectedModelChange:
    
    def __init__(self, model):
        self.model = model
    
    def compute_egl(self, X_unlabeled, X_labeled, y_labeled):
        
        self.model.fit(X_labeled, y_labeled)

        # Compute the expected gradient length for each unlabeled instance
        egl_scores = []

        for i in range(X_unlabeled.shape[0]):

            #select the unlabeled instance and predict is probability
            x = X_unlabeled.iloc[i:i+1, :]
            x = x.reset_index(drop=True)

            oij = self.model.predict_proba(x)[0]

            #get the positive gradient
            y_pos = pd.Series([1])
            gradient_w_pos, gradient_b_pos = self.model.calculate_gradient(x, y_pos, oij)
            gradient_pos = np.concatenate((gradient_w_pos.flatten(), gradient_b_pos.flatten()))

            #get the negative gradient
            y_neg = pd.Series([0])
            gradient_w_neg, gradient_b_neg = self.model.calculate_gradient(x, y_neg, oij)
            gradient_neg = np.concatenate((gradient_w_neg.flatten(), gradient_b_neg.flatten()))

            #compute the expected gradient length
            egl = oij * np.linalg.norm(gradient_pos) + (1 - oij) * np.linalg.norm(gradient_neg)
            egl_scores.append(egl[0])

        return egl_scores
    
    def query(self, X_train, y_train, X_test, batch_size):
        
        scores = self.compute_egl(X_test, X_train, y_train)
        
        scores = np.array(scores)
        highest_idx = np.argsort(scores)[::-1][:batch_size]
        
        return highest_idx

class QueryByCommittee:
    
    def __init__(self, model):
        self.model = model
        self.committee_scores = None
        self.n_estimators = None
    
    def committee_prediction(self, X_train, y_train, X_test):
        
        params = {"random_state": np.random.randint(0, 10000)}
        self.model.set_params(**params)
        
        self.model.fit(X_train.values, y_train.values)
        
        self.n_estimators = len(self.model.estimators_)
        
        self.committee_scores = np.zeros((self.n_estimators, X_test.shape[0], 2))
        
        for j, estimator in enumerate(self.model.estimators_):
            self.committee_scores[j, : ,:] = estimator.predict_proba(X_test.values)
        
    def compute_avg_kl_divergence(self):
        
        #Calculate the average KL divergence.
        probas_mean = np.mean(self.committee_scores, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.nansum(
                np.nansum(self.committee_scores * np.log(self.committee_scores / probas_mean), axis=2), axis=0
            )
        
        kl_divs = scores / self.n_estimators
        
        return kl_divs
    
    def query(self, X_train, y_train, X_test, batch_size):
        
        pred = self.committee_prediction(X_train, y_train, X_test)
        
        scores = self.compute_avg_kl_divergence()
        
        scores = np.array(scores)
        highest_idx = np.argsort(scores)[::-1][:batch_size]
        
        return highest_idx

