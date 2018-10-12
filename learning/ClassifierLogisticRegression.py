
import os
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from learning.BaseClassifier import BaseClassifier
from learning.BaseClassifier import BaseOptionsClassifier


class OptionsLogisticRegression(BaseOptionsClassifier):

    def __init__(self, dir_models_base, options_filename_dataset_training, options_clf=None):
        self.penalty = 'l2';
        self.C = 0.01;
        self.random_state = None;
        if options_clf is not None:
            if 'penalty' in options_clf.keys():
                self.penalty = options_clf['penalty'];
            if 'C' in options_clf.keys():
                self.C = options_clf['C'];
        BaseOptionsClassifier.__init__(self, 'logisticregression', dir_models_base, options_filename_dataset_training, self._getFilenameOptionsLR());


    def _getDirModel(self, dir_models_base):
        dir_models = dir_models_base + '/' + self.name + '/';
        if not os.path.exists(dir_models):
            os.makedirs(dir_models);
        return dir_models;

    def getPenalty(self):
        return self.penalty;

    def getC(self):
        return self.C;

    def getRandomState(self):
        return self.random_state;

    def _getFilenameOptionsLR(self):
        strOpt = 'penalty_' + str(self.penalty);
        strOpt = strOpt + '_C' + str(self.C);
        return strOpt


class ClassifierLogisticRegression(BaseClassifier):

    def __init__(self, options):
        BaseClassifier.__init__(self);
        self.options = options;
        self.clf = LogisticRegression(penalty=self.options.getPenalty(),
                                      C=self.options.getC(),
                                      random_state=self.options.getRandomState());
        return;

    def _getLearnedFeatures(self):
        return self.clf.coef_;

    def save(self, run):
        filename = self.options.getFilenameClf(run);
        joblib.dump(self.clf, filename);

    def loadFromFile(self, run):
        filename = self.options.getFilenameClf(run);
        BaseClassifier._loadClassifier(self, filename);

    def saveLearnedFeatures(self, run):
        learned_features = self._getLearnedFeatures()
        filename = self.options.getFilenameLearnedFeatures(run);
        self._writeNumericListToFile(learned_features.flatten(), filename);

    def getLearnedFeatures(self):
        self._getLearnedFeatures();