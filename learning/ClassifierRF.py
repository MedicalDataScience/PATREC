
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from learning.BaseClassifier import BaseClassifier


class OptionsRF():

    def __init__(self, dir_models_base, options_filename_dataset_training, options_dict=None):
        self.name = 'rf';
        self.dir_model = self._getDirModel(dir_models_base);
        self.n_estimators = 100;
        self.max_depth = 15;
        self.random_state = None;
        self.class_weight = None;
        self.n_jobs = 8;
        self.filename_options_training_data = options_filename_dataset_training;

        if options_dict is not None:
            if options_dict['n_estimators'] is not None:
                self.n_estimators = options_dict['n_estimators'];
            if options_dict['max_depth'] is not None:
                self.max_depth = options_dict['max_depth'];
            if options_dict['random_state'] is not None:
                self.random_state = options_dict['random_state'];
            if options_dict['class_weight'] is not None:
                self.class_weight = options_dict['class_weight'];
            if options_dict['n_jobs'] is not None:
                self.n_jobs = options_dict['n_jobs'];


    def _getDirModel(self, dir_models_base):
        dir_models = dir_models_base + '/' + self.name + '/';
        if not os.path.exists(dir_models):
            os.makedirs(dir_models);
        return dir_models;

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

    def getName(self):
        return self.name;

    def getFilenameOptions(self):
        strOpt = self.name;
        strOpt = strOpt + '_numEst' + str(self.n_estimators);
        strOpt = strOpt + '_maxDepth' + str(self.max_depth);
        return strOpt

    def getFilenameOptionsTrainingData(self):
        return self.filename_options_training_data;


    def getFilenameLearnedFeatures(self, run):
        filename = self.dir_model + 'learned_features_' + self.getFilenameOptions() + '_' + self.filename_options_training_data + '_run' + str(run) + '.sav';
        return filename;


    def getFilenameClf(self, run):
        filename = self.dir_model + self.getFilenameOptions() + '_' + self.filename_options_training_data + '_run' + str(run) + '.sav';
        return filename;


    def getDirModel(self):
        return self.dir_model;









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


    def _writeNumericListToFile(self, numList, filename):
        file = open(filename, 'w');
        for num in numList:
            file.write(str(num) + '\n');
        file.close();


    def save(self, run):
        filename = self.options.getFilenameClf(run);
        joblib.dump(self.clf, filename);


    def saveLearnedFeatures(self, run):
        learned_features = self._getLearnedFeatures()
        filename = self.options.getFilenameLearnedFeatures(run);
        self._writeNumericListToFile(learned_features, filename);


    def getLearnedFeatures(self):
        self._getLearnedFeatures();