

from sklearn.linear_model import SGDClassifier

from learning.BaseClassifier import BaseClassifier
from learning.BaseClassifier import BaseOptionsClassifier

class OptionsSGD(BaseOptionsClassifier):

    def __init__(self, dir_models_base, options_filename_dataset_training, options_clf=None):
        self.loss = 'log';
        self.penalty = 'l1';
        if options_clf is not None:
            if 'loss' in options_clf.keys():
                self.loss = options_clf['loss'];
            if 'penalty' in options_clf.keys():
                self.penalty = options_clf['penalty'];

        BaseOptionsClassifier.__init__(self, 'sgd', dir_models_base, options_filename_dataset_training, self._getFilenameOptionsSGD());
        return;


    def getLoss(self):
        return self.loss;

    def getPenalty(self):
        return self.penalty;

    def _getFilenameOptionsSGD(self):
        strOpt = 'loss' + str(self.loss);
        strOpt = strOpt + '_penalty' + str(self.penalty);
        return strOpt



class ClassifierSGD(BaseClassifier):

    def __init__(self, options):
        BaseClassifier.__init__(self);
        self.options = options;
        self.clf = SGDClassifier(loss=self.options.getLoss(),
                                 penalty=self.options.getPenalty(),
                                 alpha=0.0001);
        return;