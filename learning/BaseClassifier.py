
import sklearn.metrics as metrics
from sklearn.externals import joblib
from sklearn import preprocessing

from utils.Results import ResultsSingleRun



class BaseClassifier:

    def __init__(self):
        self.clf = None;
        return;


    def __loadClassifier(self, filename_classifier):
        clf_model = joblib.load(filename_classifier)
        self.clf = clf_model;


    def train(self, df):
        labels = df['Wiederkehrer'].values;
        data = df.drop('Wiederkehrer', axis=1).values;
        self.clf.fit(data, labels);


    def predict(self, df):
        labels = df['Wiederkehrer'].values;
        data = df.drop('Wiederkehrer', axis=1).values;

        predictions = self.clf.predict_proba(data);
        fpr, tpr, thresholds_fprtpr = metrics.roc_curve(labels, predictions[:, 1]);
        precision, recall, thresholds_pr = metrics.precision_recall_curve(labels, predictions[:, 1]);
        average_precision = metrics.average_precision_score(labels, predictions[:, 1]);
        roc_auc = metrics.auc(fpr, tpr);

        results = ResultsSingleRun();
        results.precision = precision;
        results.recall = recall;
        results.thresholds_precision_recall = thresholds_pr;
        results.average_precision = average_precision;
        results.tpr = tpr;
        results.fpr = fpr;
        results.thresholds_precision_recall = thresholds_fprtpr;
        results.roc_auc = roc_auc;
        results.calcFMeasure(precision, recall);
        return results;

