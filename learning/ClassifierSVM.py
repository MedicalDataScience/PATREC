
import numpy as np

from sklearn.svm import SVC
from sklearn.externals import joblib

from learning.BaseClassifier import BaseOptionsClassifier
from learning.BaseClassifier import BaseClassifier

class OptionsSVM(BaseOptionsClassifier):

    def __init__(self, dir_models_base, options_filename_dataset_training, options_clf=None):
        self.kernel = 'rbf';
        self.C = 1.0;

        if options_clf is not None:
            if 'kernel' in options_clf.keys():
                self.kernel = options_clf['kernel'];
            if 'C' in options_clf.keys():
                self.C = options_clf['C'];

        BaseOptionsClassifier.__init__(self, 'svm', dir_models_base, options_filename_dataset_training,self._getFilenameOptionsSVM());
        return;


    def getKernelMethod(self):
        return self.kernel;

    def getC(self):
        return self.C;

    def _getFilenameOptionsSVM(self):
        strOpt = str(self.kernel);
        strOpt = strOpt + '_C' + str(self.C);
        return strOpt



class ClassifierSVM(BaseClassifier):

    def __init__(self, options):
        BaseClassifier.__init__(self);
        self.options = options;
        self.clf = SVC(kernel=self.options.getKernelMethod(),
                       C=self.options.getC(),
                       probability=True);


    def _getLearnedFeatures(self):
        return np.mean(self.clf.support_vectors_, axis=0);


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
