import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from itertools import cycle
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    roc_curve, matthews_corrcoef, auc, confusion_matrix)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=16)   # fontsize of the figure title
sns.set_context('paper')


class CreateBaselines(object):
    """
    Creating baseline models simplified


    Arguments:
        - X_train: training input
        - X_test: test input
        - y_train: training labels
        - y_test: test labels
        - n_jobs: how many jobs to run at the same time (multiprocessing)
        - cv: number of folds in cross validation

    """
    def __init__(self, X_train, y_train, X_test, y_test, n_jobs=1, cv=5):
        self.X_test = X_test
        self.X_train = X_train
        self.y_test = y_test
        self.y_train = y_train
        self.n_jobs = n_jobs
        self.cv = cv

        self.sgd_param_grid = {
            'penalty': ['l2'],
            'alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'tol': [1e+1, 1e-0, 1e-1, 1e-2, 1e-3, 1e-4],
            'loss': ['modified_huber'],
            'max_iter': [10000]
            }

        self.rf_param_grid = {
            'n_estimators': [10, 50, 100, 200, 500, 1000],
            'max_features': ['auto', 'sqrt'],
            'min_samples_split': [2, 5, 10],
            'bootstrap': [True, False]
            }

        self.log_param_grid = {
            'C': [1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'max_iter': [10000],
            'tol': [1e+1, 1e-0, 1e-1, 1e-2, 1e-3, 1e-4]
            }

        self.type2net = {
            'Logistic': LogisticRegression(),
            'Linear': SGDClassifier(),
            'Random forest': RandomForestClassifier()
            }
        self.type2grid = {
            'Logistic': self.log_param_grid,
            'Linear': self.sgd_param_grid,
            'Random forest': self.rf_param_grid}
        self.type2attr = {
            'Logistic': 'baseline_log',
            'Linear': 'baseline_linear',
            'Random forest': 'baseline_rf'
            }
        self.type2label = {
            'Logistic': 'Logistic baseline',
            'Linear': 'SGD linear baseline',
            'Random forest': 'Random forest baseline'
            }

        # keep track of which baseline exists in self
        self.exists = []

    def create_baseline(self, model_type, evaluate=True, file_name=None):
        """
        Create a baseline model by doing parameter optimization with cross validation

        Arguments:
            - model_type: a string telling which model type to create. Possible models: Linear (support vector machine), Logistic (logistic regression) and Random Forest.
            - evaluate: whether or not to evaluate the best classifier by creating confusion matrice, ROC curve and feature importance plots.
            - file_name: if given, export all plots to that file.
        """

        if model_type not in self.exists:
            self.exists.append(model_type)
        else:
            warnings.warn('Baseline already exists, but will be replaced')

        model = GridSearchCV(
            self.type2net[model_type],
            self.type2grid[model_type],
            cv=self.cv, verbose=1, n_jobs=self.n_jobs)
        model.fit(self.X_train, self.y_train)

        setattr(self, self.type2attr[model_type], model)

        if evaluate:
            label = self.type2label[model_type]
            cm_fig = self.plot_cm(model=model, label=label)
            features_fig = self.plot_features(model=model, label=label)
            roc_fig = self.plot_roc_auc(model=model, label=label)

            return cm_fig, features_fig, roc_fig

    def plot_cm(self, model, label):
        """Plot confusion matrix"""
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(y_true=self.y_test, y_pred=y_pred)
        svm = sns.heatmap(
            cm, annot=True, fmt='.3f',
            linewidths=.5, square=True, cmap='Blues_r'
            )
        fig = svm.get_figure()
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.title("{}:\nMCC: {:.3f}".format(
            label, matthews_corrcoef(self.y_test, y_pred))
            )
        plt.tight_layout()
        plt.close()
        return fig

    def plot_features(self, model, label):
        """Create feature importance plots"""

        features = self.X_train.columns
        try:
            coefficients = model.best_estimator_.coef_[0]
            ylabel = 'coefficient'
        except AttributeError:
            coefficients = model.best_estimator_.feature_importances_
            indices = np.argsort(features)
            features = features[indices]
            ylabel = 'importance'

        fig = plt.figure(figsize=(10, 10))
        plt.bar(np.arange(len(coefficients)), coefficients)
        plt.xticks(np.arange(len(coefficients)), features, rotation='vertical')
        plt.ylabel(ylabel)
        plt.title("{}:\nFeature importance".format(label))
        plt.tight_layout()
        plt.close()
        return fig

    def plot_roc_auc(self, model, label):
        """Plot the ROC curve with AUC"""
        y_score = model.predict_proba(self.X_test)

        fpr, tpr, _ = roc_curve(self.y_test, y_score[:, 1])
        score_auc = auc(fpr, tpr)

        fig = plt.figure()
        plt.plot(fpr, tpr, label='{} (area = {:.2f})'.format(label, score_auc))
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('{}:\nReceiver operating characteristic'.format(label))
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.close()
        return fig

    def plot_multiple_roc(self):
        """Compare ROC curves for all trained classifiers in the object"""
        if len(self.exists) == 0:
            raise ValueError("Please create some baselines first!")

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_true = self.y_test

        for model_type in self.exists:
            model = getattr(self, self.type2attr[model_type])
            y_score = model.predict_proba(self.X_test)

            if y_score.shape[1] >= 2:
                y_score = y_score[:, 1]

            fpr[model_type], tpr[model_type], _ = roc_curve(
                y_true=y_true,
                y_score=y_score
            )
            roc_auc[model_type] = auc(fpr[model_type], tpr[model_type])

        fig = plt.figure(figsize=(10, 10))
        lw = 2
        colors = cycle([
            'blue', 'green', 'red', 'orange',
            'purple', 'pink', 'aqua', 'darkorange',
            'yellow', 'brown'
            ])

        for i, color in zip(self.exists, colors):
            plt.plot(
                fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of model {0} (area = {1:0.2f})'.format(
                    self.type2label[i], roc_auc[i])
                    )

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curves per model type')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.close()
        return fig

    def export_all_baselines(self, file_name):
        """Export all plots for all baselines to a file"""

        if len(self.exists) == 0:
            raise ValueError("Please create some baselines first!")

        with PdfPages(file_name) as pdf:

            for model_type in self.exists:
                model = getattr(self, self.type2attr[model_type])
                label = self.type2label[model_type]

                cm_fig = self.plot_cm(model=model, label=label)
                features_fig = self.plot_features(model=model, label=label)
                roc_fig = self.plot_roc_auc(model=model, label=label)

                pdf.savefig(cm_fig)
                pdf.savefig(features_fig)
                pdf.savefig(roc_fig)

            roc_all = self.plot_multiple_roc()
            pdf.savefig(roc_all)
            plt.close('all')