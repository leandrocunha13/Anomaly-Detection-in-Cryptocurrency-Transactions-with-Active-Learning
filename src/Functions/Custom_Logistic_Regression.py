import numpy as np
import copy


class Custom_LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, tol = 0.0001):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.tol = tol
        self.w = None
        self.b = None
        self.costs = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, dim):
        self.w = np.zeros((dim,1))
        self.b = 0
    
    def compute_cost(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.w) + self.b
        a = self.sigmoid(z)
        cost = (-1/m) * np.sum(y * np.log(a + 1e-9) + (1-y) * np.log(1-a + 1e-9))
        return cost
    
    def calculate_gradient(self, X, y, pred):
        m = X.shape[0]
        dw = (1/m) * np.dot(X.T, (pred - y))
        db = (1/m) * np.sum(pred - y)
        
        return dw, db
    
    def fit(self, X, y):
        X = self._transform_x(X)
        y = self._transform_y(y)

        self.initialize_parameters(X.shape[1])
        
        m = X.shape[0]
        
        best_cost = float('inf')  # Initialize with a high value

        for i in range(self.num_iterations):
            z = np.dot(X, self.w) + self.b
            pred = self.sigmoid(z)
            
            #compute gradient
            dw, db = self.calculate_gradient(X, y, pred)
            
            #update model parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            #compute cost/loss
            cost = self.compute_cost(X, y)
            self.costs.append(cost)
            
            if i % 100 == 0:
                print(f'Cost after iteration {i}: {cost}')     
            
            if cost < best_cost - self.tol:
                best_cost = cost
            else:
                break      
    
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        predictions = self.sigmoid(z)
        return predictions.round()
    
    def predict_proba(self, X):
        z = np.dot(X, self.w) + self.b
        predictions = self.sigmoid(z)
        return predictions
    
    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x.values

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.values.reshape(y.shape[0], 1)
