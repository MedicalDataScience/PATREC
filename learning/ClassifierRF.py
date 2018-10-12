
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from learning.BaseClassifier import BaseClassifier
from learning.BaseClassifier import BaseOptionsClassifier

class OptionsRF(BaseOptionsClassifier):

    def __init__(self, dir_models_base, options_filename_dataset_training, options_clf=None):
        self.n_estimators = 100;
        self.max_depth = 15;
        self.random_state = None;
        self.class_weight = None;
        self.n_jobs = 8;

        if options_clf is not None:
            if 'n_estimators' in options_clf.keys():
                self.n_estimators = options_clf['n_estimators'];
            if 'max_depth' in options_clf.keys():
                self.max_depth = options_clf['max_depth'];
            if 'random_state' in options_clf.keys():
                self.random_state = options_clf['random_state'];
            if 'class_weight' in options_clf.keys():
                self.class_weight = options_clf['class_weight'];
            if 'n_jobs' in options_clf.keys():
                self.n_jobs = options_clf['n_jobs'];
        BaseOptionsClassifier.__init__(self, 'rf', dir_models_base, options_filename_dataset_training, self._getFilenameOptionsRF());
        return;


    def getNumEstimators(self):
        return self.n_estimators;

    def getMaxDepth(self):
        return self.max_depth;

    def getRandomState(self):
        return self.random_state;

    def getClassWeight(self):
        return self.class_weight;

    def getNumJobs(self):
        return self.n_jobs;

    def _getFilenameOptionsRF(self):
        strOpt = 'numEst' + str(self.n_estimators);
        strOpt = strOpt + '_maxDepth' + str(self.max_depth);
        return strOpt



class ClassifierRF(BaseClassifier):

    def __init__(self, options):
        BaseClassifier.__init__(self);
        self.options = options;
        self.clf = RandomForestClassifier(n_estimators=self.options.getNumEstimators(),
                                          max_depth=self.options.getMaxDepth(),
                                          random_state=self.options.getRandomState(),
                                          n_jobs=self.options.getNumJobs(),
                                          class_weight=self.options.getClassWeight());

        return;


    def _getLearnedFeatures(self):
        return self.clf.feature_importances_;


    def save(self, run):
        filename = self.options.getFilenameClf(run);
        joblib.dump(self.clf, filename);


    def loadFromFile(self, run):
        filename = self.options.getFilenameClf(run);
        BaseClassifier._loadClassifier(self, filename);


    def saveLearnedFeatures(self, run):
        learned_features = self._getLearnedFeatures()
        filename = self.options.getFilenameLearnedFeatures(run);
        self._writeNumericListToFile(learned_features, filename);


    def getLearnedFeatures(self):
        self._getLearnedFeatures();