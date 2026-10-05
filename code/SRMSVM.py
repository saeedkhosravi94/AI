import numpy as np
from scipy.optimize import minimize

class SRMSVM:

    
    def __init__(self, C=1.0):
        self.C = C  
        self.w = None
        self.b = None
        
    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        
        unique_labels = np.unique(y)
        if not (set(unique_labels) <= {1.0, -1.0}):
            raise ValueError("Labels must be +1 or -1")
        
        n_samples, n_features = X.shape
        
        self.w = np.zeros((n_features, 1))
        self.b = 0.0
        
        self._optimize_srm(X, y)
        
        return self
    
    def _optimize_srm(self, X, y):
        """Solve the 1-norm SVM problem (RLP formulation)"""
        n_samples, n_features = X.shape
        
        def objective(params):
            w = params[:n_features].reshape(-1, 1)
            b = params[n_features]
            s = params[n_features+1:n_features+1+n_features]  
            eta = params[n_features+1+n_features:]  
            
            misclassification_cost = self.C * np.sum(eta)
            regularization = np.sum(s)  
            
            return misclassification_cost + regularization
        
        def constraints(params):
            w = params[:n_features]
            b = params[n_features]
            eta = params[n_features+1+n_features:]
            
            return y.flatten() * (X.dot(w) - b) + eta - 1
        
        bounds = (
            [(None, None)] * n_features +  
            [(None, None)] +  
            [(0, None)] * n_features +  
            [(0, None)] * n_samples  
        )
        
        x0 = np.zeros(n_features + 1 + n_features + n_samples)
        x0[n_features] = 0.1 
        
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraints}
        )
        
        self.w = res.x[:n_features].reshape(-1, 1)
        self.b = res.x[n_features]
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        if self.w is None or self.b is None:
            raise ValueError("Model not fitted yet")
            
        X = np.asarray(X, dtype=np.float64)
        scores = X.dot(self.w) + self.b
        return np.where(scores >= 0, 1, -1).ravel()