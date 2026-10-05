import numpy as np
from scipy.optimize import minimize

class S3VM_Unconstrained:
    
    def __init__(self, C=1.0, eps=1e-4):
        self.C = C  
        self.eps = eps
        self.w = None
        self.b = None
        
    def fit(self, X_labeled, y_labeled, X_unlabeled):
        X_labeled = np.asarray(X_labeled, dtype=np.float64)
        y_labeled = np.asarray(y_labeled, dtype=np.float64).reshape(-1, 1)
        X_unlabeled = np.asarray(X_unlabeled, dtype=np.float64)
        
        unique_labels = np.unique(y_labeled)
        if not (set(unique_labels) <= {1.0, -1.0}):
            raise ValueError("Labels must be +1 or -1")
        
        n_features = X_labeled.shape[1]
        self.w = np.zeros((n_features, 1))
        self.b = 0.0
        
        X_labeled_aug = np.hstack([X_labeled, np.ones((X_labeled.shape[0], 1))])
        X_unlabeled_aug = np.hstack([X_unlabeled, np.ones((X_unlabeled.shape[0], 1))])
        
        
        unlabeled_scores = X_unlabeled_aug @ np.vstack([self.w, self.b])
        y_unlabeled = np.sign(unlabeled_scores)
        y_unlabeled[y_unlabeled == 0] = 1 
        
        X_aug = np.vstack([X_labeled_aug, X_unlabeled_aug])
        y = np.vstack([y_labeled, y_unlabeled])
        
        self._optimize(X_aug, y)
        
        new_scores = X_unlabeled_aug @ np.vstack([self.w, self.b])
        if np.all(np.sign(new_scores) == y_unlabeled):
            return 
                
        return self
    
    def _optimize(self, X_aug, y):
        _, n_features = X_aug.shape
        
        def objective(params):
            w = params[:-1].reshape(-1, 1)
            b = params[-1]
            margins = y * (X_aug[:, :-1] @ w + X_aug[:, -1] * b)
            
            hinge_loss = np.sum(np.maximum(0, 1 - margins))
            
            norm1_w = np.sum(np.abs(w))
            
            return self.C * hinge_loss + norm1_w
        
        x0 = np.zeros(n_features)
        x0[-1] = 0 
        
        bounds = [(None, None) if i == n_features-1 else (None, None) 
                 for i in range(n_features)]
        
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        self.w = res.x[:-1].reshape(-1, 1)
        self.b = res.x[-1]
    
    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("Model not fitted yet")
            
        X = np.asarray(X, dtype=np.float64)
        scores = X @ self.w + self.b
        return np.where(scores >= 0, 1, -1).ravel()