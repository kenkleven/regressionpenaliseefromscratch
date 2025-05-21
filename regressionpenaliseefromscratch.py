import numpy as np


class RegressionRidge():
    def __init__(self, alpha=0.01, max_iter=100, lmbd=0.1):
        self.alpha = alpha
        self.max_iter = max_iter
        self.lmbd = lmbd
        self.w = None
        self.b = 0.0


    def pred(self, X):
        return np.dot(X, self.w) + self.b

    def cout(self, X, y):
        y_pred = self.pred(X)
        mse = np.mean((y_pred - y) ** 2)
        ridge_penalty = self.lmbd * np.sum(self.w ** 2)
        return mse + ridge_penalty

    def gradients(self, X, y):
        m = len(y)
        y_pred = self.pred(X)
        error = y_pred - y
        dw = (2/m) * np.dot(X.T, error) + 2 * self.lmbd * self.w
        db = (2/m) * np.sum(error)
        return dw, db

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0

        for i in range(self.max_iter):
            dw, db = self.gradients(X, y)
            self.w -= self.alpha * dw
            self.b -= self.alpha * db

            if i % 10 == 0 or i == self.max_iter - 1:
                cost = self.cout(X, y)
                print(f"Époque {i} : coût = {cost:.4f}, w = {self.w}, b = {self.b:.4f}")

    def get_params(self):
        return self.w, self.b
    

class RegressionLasso():
    def __init__(self, alpha=0.01, max_iter=100, lmbd=0.1):
        self.alpha = alpha
        self.max_iter = max_iter
        self.lmbd = lmbd
        self.w = None
        self.b = 0.0

    def pred(self, X):
        return np.dot(X, self.w) + self.b

    def cout(self, X, y):
        y_pred = self.pred(X)
        mse = np.mean((y_pred - y) ** 2)
        lasso_penalty = self.lmbd * np.sum(np.abs(self.w))
        return mse + lasso_penalty

    def gradients(self, X, y):
        m = len(y)
        y_pred = self.pred(X)
        error = y_pred - y

        dw = (2/m) * np.dot(X.T, error) + self.lmbd * np.sign(self.w)
        db = (2/m) * np.sum(error)
        return dw, db

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0

        for i in range(self.max_iter):
            dw, db = self.gradients(X, y)
            self.w -= self.alpha * dw
            self.b -= self.alpha * db

            if i % 10 == 0 or i == self.max_iter - 1:
                cost = self.cout(X, y)
                print(f"Époque {i} : coût = {cost:.4f}, w = {self.w}, b = {self.b:.4f}")

    def get_params(self):
        return self.w, self.b
