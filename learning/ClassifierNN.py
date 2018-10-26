

from learning.BaseClassifier import BaseClassifier
from learning.BaseClassifier import BaseOptionsClassifier

class OptionsNN(BaseOptionsClassifier):

    def __init__(self, dir_models_base, options_filename_dataset_training, options_clf=None):
        self.hidden_units = [20, 10, 10];
        self.learningrate = 0.1;
        self.dropout = 0.3;
        self.batchnorm = True
        self.batch_size = 1280;
        self.training_epochs = 250;
        self.pretrained = None;

        if options_clf is not None:
            keys = list(options_clf.keys())
            if 'hidden_units' in keys:
                self.hidden_units = options_clf['hidden_units'];
            if 'learningrate' in keys:
                self.learning_rate = options_clf['learningrate'];
            if 'dropout' in keys:
                self.dropout = options_clf['dropout'];
            if 'batch_size' in keys:
                self.batch_size = options_clf['batch_size'];
            if 'training_epochs' in keys:
                self.training_epochs = options_clf['training_epochs'];
            if 'batchnorm' in keys:
                self.batchnorm = options_clf['batchnorm'];
            if 'pretrained' in keys:
                self.pretrained = options_clf['pretrained'];
        BaseOptionsClassifier.__init__(self, 'nn', dir_models_base, options_filename_dataset_training, self._getFilenameOptionsNN());
        return;


    def _getFilenameOptionsNN(self):
        if self.pretrained is not None:
            strOpt = self.pretrained + '_';
        else:
            strOpt = '';
        strOpt = strOpt + str(self.hidden_units[0]);
        for k in range(1, len(self.hidden_units)):
            strOpt = strOpt + '_' + str(self.hidden_units[k]);

        strOpt = strOpt + '_dropout_' + str(self.dropout);
        strOpt = strOpt + '_learningrate_' + str(self.learningrate);
        strOpt = strOpt + '_batchnorm_' + str(self.batchnorm);
        strOpt = strOpt + '_batchsize_' + str(self.batch_size);
        return strOpt;



class ClassifierNN(BaseClassifier):
    def __init__(self, options):
        BaseClassifier.__init__(self);
        self.options = options;
