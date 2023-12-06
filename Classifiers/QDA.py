import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

class QDABaggingWithCrossValidationGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy", cv_range=None):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv_range = cv_range if cv_range is not None else [5, 10, 15]  # Default CV range
        self.qda = QuadraticDiscriminantAnalysis()
        self.accuracy = None  # Store accuracy

        # Define parameter grid for grid search
        self.param_grid = {
            'reg_param': [0.0, 0.1, 0.2],  # Customize the list according to your needs
        }

    def find_best_num_folds(self, X, y):
        min_samples_per_class = min(np.bincount(y))
        best_num_folds = min(min_samples_per_class, max(self.cv_range))
        return best_num_folds

    def fit(self, X, y, cv=None):
        # Define best_num_folds here if cv is not provided
        if cv is None:
            cv = self.find_best_num_folds(X, y)

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.qda, X, y, cv=skf, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = np.mean(scores)

    def get_accuracy(self):
        return self.accuracy

    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        # Use best_num_folds if cv is not provided
        if cv is None:
            cv = self.find_best_num_folds(X, y)

        skf = StratifiedKFold(n_splits=cv, shuffle=True)
        train_sizes, train_scores, test_scores = learning_curve(
            self.qda, X, y, train_sizes=train_sizes, cv=skf, n_jobs=self.n_jobs, scoring=self.scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)


        plt.figure(figsize=(10, 6))

        # Plot mean training and test scores
        plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement", color="r")
        plt.plot(train_sizes, test_scores_mean, label="Score de Test", color="g")

        # Add shaded regions for the variability in training and test scores
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("QDA")
        plt.legend()
        plt.grid()

    def grid_search(self, X, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True)  # Use StratifiedKFold for grid search
        grid_search = GridSearchCV(self.qda, self.param_grid, cv=skf, n_jobs=self.n_jobs, scoring=self.scoring)
        grid_search.fit(X, y)
        self.qda = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_
 
class PCAandQDABaggingWithCrossValidationGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy", cv_range=None):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv_range = cv_range if cv_range is not None else [5, 10, 15]  # Default CV range
        self.pca = PCA()
        self.qda = QuadraticDiscriminantAnalysis()
        self.accuracy = None  # Store accuracy

        # Define parameter grid for PCA
        self.pca_param_grid = {
            'n_components': [10, 20, 30, 40, 60],  # Customize the list according to your needs
        }

        # Define parameter grid for QDA
        self.qda_param_grid = {
            'reg_param': [0.0, 0.1, 0.2],  # Customize the list according to your needs
        }

    def find_best_num_folds(self, X, y):
        min_samples_per_class = min(np.bincount(y))
        best_num_folds = min(min_samples_per_class, max(self.cv_range))
        return best_num_folds

    def fit(self, X, y, cv=None):
        # Perform PCA and transform data
        self.pca.fit(X)
        X_pca = self.pca.transform(X)

        # Use best_num_folds if cv is not provided
        if cv is None:
            cv = self.find_best_num_folds(X, y)

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.qda, X_pca, y, cv=skf, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = np.mean(scores)

    def get_accuracy(self):
        return self.accuracy

    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        # Use best_num_folds if cv is not provided
        if cv is None:
            cv = self.find_best_num_folds(X, y)

        skf = StratifiedKFold(n_splits=cv, shuffle=True)
        train_sizes, train_scores, test_scores = learning_curve(
            self.qda, X, y, train_sizes=train_sizes, cv=skf, n_jobs=self.n_jobs, scoring=self.scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)


        plt.figure(figsize=(10, 6))

        # Plot mean training and test scores
        plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement", color="r")
        plt.plot(train_sizes, test_scores_mean, label="Score de Test", color="g")

        # Add shaded regions for the variability in training and test scores
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("QDA")
        plt.legend()
        plt.grid()

    def grid_search(self, X, y, num_folds=None):
        # Use the provided number of folds or find the best number of folds
        if num_folds is None:
            num_folds = self.find_best_num_folds(X, y)

        # Grid search for PCA
        pca_grid_search = GridSearchCV(self.pca, self.pca_param_grid, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        pca_grid_search.fit(X, y)
        best_pca = pca_grid_search.best_estimator_

        # Apply PCA transformation
        X_pca = best_pca.transform(X)

        # Grid search for QDA
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        qda_grid_search = GridSearchCV(self.qda, self.qda_param_grid, cv=skf, n_jobs=self.n_jobs, scoring=self.scoring)
        qda_grid_search.fit(X_pca, y)
        best_qda = qda_grid_search.best_estimator_

        return best_pca, best_qda, pca_grid_search.best_params_, pca_grid_search.best_score_, qda_grid_search.best_params_, qda_grid_search.best_score_

class ForwardSelectionQDA:
    def __init__(self, n_jobs=-1, scoring="accuracy", cv_range=None):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv_range = cv_range if cv_range is not None else [5, 10, 15]  # Default CV range
        self.qda = QuadraticDiscriminantAnalysis()
        self.best_accuracy = None
        self.best_num_features = None
 
        # Define parameter grid for QDA
        self.qda_param_grid = {
            'reg_param': [0.0, 0.1, 0.2],  # Customize the list according to your needs
        }
 
    def find_best_num_folds(self, X, y):
        min_samples_per_class = min(np.bincount(y))
        best_num_folds = min(min_samples_per_class, max(self.cv_range))
        return best_num_folds

 
    def forward_selection(self, X, y, max_features):
        best_accuracy = 0.0
        best_num_features = 1
        best_X_selected = X[:, :1]
 
        for num_features in range(1, max_features + 1):
            # Use the 'num_features' first columns as features
            X_selected = X[:, :num_features]
 
            # Adjust the QDA model
            self.qda.fit(X_selected, y)
 
            # Perform cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            scores = cross_val_score(self.qda, X_selected, y, cv=skf, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()
 
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_features = num_features
                best_X_selected = X_selected
 
        return best_X_selected, best_num_features, best_accuracy
 
    def fit(self, X, y):
        # Determine the best number of features to use
        X_selected, best_num_features, best_accuracy = self.forward_selection(X, y, X.shape[1])
 
        # Adjust the QDA model with the best number of features
        self.qda.fit(X_selected, y)
        self.best_num_features = best_num_features
        self.best_accuracy = best_accuracy
 
    def get_accuracy(self):
        return self.best_accuracy
 
    def get_best_num_features(self):
        return self.best_num_features
 
    def plot_learning_curve(self, X, y, train_sizes, cv=None, save_path=None):
        # Use best_num_folds if cv is not provided
        if cv is None:
            cv = self.find_best_num_folds(X, y)

        # Ensure that cv is not greater than the number of samples in the smallest class
        cv = min(cv, min(np.bincount(y)))

        skf = StratifiedKFold(n_splits=cv, shuffle=True)
        train_sizes, train_scores, test_scores = learning_curve(
            self.qda, X, y, train_sizes=train_sizes, cv=skf, n_jobs=self.n_jobs, scoring=self.scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)


        plt.figure(figsize=(10, 6))

        # Plot mean training and test scores
        plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement", color="r")
        plt.plot(train_sizes, test_scores_mean, label="Score de Test", color="g")

        # Add shaded regions for the variability in training and test scores
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("QDA")
        plt.legend()
        plt.grid()
 
    def grid_search(self, X, y):
        # Grid search for QDA
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        grid_search = GridSearchCV(self.qda, self.qda_param_grid, cv=skf, n_jobs=self.n_jobs, scoring=self.scoring)
        grid_search.fit(X, y)
        self.qda = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_
