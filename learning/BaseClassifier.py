
import os
import sklearn.metrics as metrics
from sklearn.externals import joblib

from utils.Results import ResultsSingleRun


class BaseOptionsClassifier:

    def __init__(self, name, dir_models_base, options_filename_dataset_training, filename_options_clf):
        self.name = name;
        self.dir_model = self._getDirModel(dir_models_base)
        self.filename_options_training_data = options_filename_dataset_training;
        self.filename_options_clf = filename_options_clf;
        return;

    def _getDirModel(self, dir_models_base):
        dir_models = dir_models_base + '/' + self.name + '/';
        if not os.path.exists(dir_models):
            os.makedirs(dir_models);
        return dir_models;

    def getFilenameOptionsTrainingData(self):
        return self.filename_options_training_data;

    def getFilenameLearnedFeatures(self, run):
        filename = self.dir_model + 'learned_features_' + self.name + '_' + self.filename_options_clf + '_' + self.filename_options_training_data + '_run' + str(run) + '.sav';
        return filename;

    def getFilenameClf(self, run):
        filename = self.dir_model + self.name + '_' + self.filename_options_clf + '_' + self.filename_options_training_data + '_run' + str(run) + '.sav';
        return filename;

    def getDirModel(self):
        return self.dir_model;

    def getName(self):
        return self.name;

    def getFilenameOptions(self):
        strOpt = self.name + '_' + self.filename_options_clf;
        return strOpt;



class BaseClassifier:

    def __init__(self):
        self.clf = None;
        return;


    def _loadClassifier(self, filename_classifier):
        clf_model = joblib.load(filename_classifier)
        self.clf = clf_model;


    def train(self, df, early_readmission_flagname):
        print('training data: ' + str(df.shape))
        labels = df[early_readmission_flagname].values;
        data = df.drop(early_readmission_flagname, axis=1).values;
        self.clf.fit(data, labels);


    def train_partial(self, df, early_readmission_flagname):
        print('training data: ' + str(df.shape))
        labels = df[early_readmission_flagname].values;
        data = df.drop(early_readmission_flagname, axis=1).values;
        self.clf.partial_fit(data, labels);


    def setResults(self, predictions, labels):
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

    def predict(self, df, early_readmission_flagname):
        print('prediction data: ' + str(df.shape))
        labels = df[early_readmission_flagname].values;
        data = df.drop(early_readmission_flagname, axis=1).values;
        predictions = self.clf.predict_proba(data);
        results = self.setResults(predictions, labels);
        return results;


    def _writeNumericListToFile(self, numList, filename):
        file = open(filename, 'w');
        for num in numList:
            file.write(str(num) + '\n');
        file.close();


