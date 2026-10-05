import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin

class S3VM_Constrained(BaseEstimator, ClassifierMixin):

    
    def __init__(self, C = 1.0, M=1e5, eps=1e-4, max_iter=100):

        self.C = C
        self.M = M
        self.eps = eps
        self.max_iter = max_iter
        self.w = None
        self.b = None
        
    def fit(self, X_labeled, y_labeled, X_unlabeled):

        X_labeled = np.asarray(X_labeled, dtype=np.float64)
        y_labeled = np.asarray(y_labeled, dtype=np.float64).reshape(-1, 1)
        X_unlabeled = np.asarray(X_unlabeled, dtype=np.float64)
        
        unique_labels = np.unique(y_labeled)
        if not (set(unique_labels) <= {1.0, -1.0}):
            raise ValueError("Labels must be +1 or -1")
            
        n_labeled, n_features = X_labeled.shape
        n_unlabeled = X_unlabeled.shape[0]
        
        self._initialize_parameters(n_features, n_labeled, n_unlabeled)
        
        X = np.vstack([X_labeled, X_unlabeled])
        
        for iteration in range(self.max_iter):
            y_unlabeled = self._predict_unlabeled(X_unlabeled)
            
            self._optimize_mip(X_labeled, y_labeled, X_unlabeled, y_unlabeled)
            
            new_labels = self._predict_unlabeled(X_unlabeled)
            if np.mean(new_labels != y_unlabeled) < self.eps:
                break
                
        return self
    
    def _initialize_parameters(self, n_features, n_labeled, n_unlabeled):
     
        self.w = np.random.normal(0, 0.01, (n_features, 1))
        self.b = 0.0
        self.eta = np.zeros(n_labeled)  
        self.xi = np.zeros(n_unlabeled)  
        self.z = np.zeros(n_unlabeled)   
        self.d = np.random.rand(n_unlabeled)  
        
    def _predict_unlabeled(self, X_unlabeled):

        scores = X_unlabeled @ self.w + self.b
        return np.where(scores >= 0, 1, -1)
    
    def _optimize_mip(self, X_labeled, y_labeled, X_unlabeled, y_unlabeled):
        
        n_labeled, n_features = X_labeled.shape
        n_unlabeled = X_unlabeled.shape[0]
        
        x0 = np.concatenate([
            self.w.flatten(),
            [self.b],
            self.eta,
            self.xi,
            self.z,
            self.d
        ])
        
        bounds = (
            [(None, None)] * n_features + 
            [(None, None)] +  
            [(0, None)] * n_labeled +  
            [(0, None)] * n_unlabeled +  
            [(0, None)] * n_unlabeled + 
            [(0, 1)] * n_unlabeled  
        )
        
        constraints = [
            {
             'type': 'ineq',
             'fun': lambda x: y_labeled.flatten() * 
                    (X_labeled @ x[:n_features] + x[n_features]) + 
                    x[n_features+1:n_features+1+n_labeled] - 1
                    
            },
            
            {
             'type': 'ineq',
             'fun': lambda x: (X_unlabeled @ x[:n_features] - x[n_features] + 
                    x[n_features+1+n_labeled:n_features+1+n_labeled+n_unlabeled] + 
                    self.M*(1 - x[-n_unlabeled:])) - 1
            },
                    
            {
                'type': 'ineq',
             'fun': lambda x: (-(X_unlabeled @ x[:n_features] - x[n_features]) + 
                    x[n_features+1+n_labeled+n_unlabeled:n_features+1+n_labeled+2*n_unlabeled] + 
                    self.M*x[-n_unlabeled:]) - 1
            }
        ]
        
        def objective(x):
            w = x[:n_features]
            eta = x[n_features+1:n_features+1+n_labeled]
            xi = x[n_features+1+n_labeled:n_features+1+n_labeled+n_unlabeled]
            z = x[n_features+1+n_labeled+n_unlabeled:n_features+1+n_labeled+2*n_unlabeled]
            
            return (self.C * (np.sum(eta) + np.sum(xi + z)) + np.sum(np.abs(w)))
                
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        self.w = res.x[:n_features].reshape(-1, 1)
        self.b = res.x[n_features]
        self.eta = res.x[n_features+1:n_features+1+n_labeled]
        self.xi = res.x[n_features+1+n_labeled:n_features+1+n_labeled+n_unlabeled]
        self.z = res.x[n_features+1+n_labeled+n_unlabeled:n_features+1+n_labeled+2*n_unlabeled]
        self.d = res.x[-n_unlabeled:]
        
    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("Model not fitted yet")
            
        X = np.asarray(X, dtype=np.float64)
        scores = X @ self.w + self.b
        return np.where(scores >= 0, 1, -1)