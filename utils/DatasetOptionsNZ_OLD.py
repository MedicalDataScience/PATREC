import sys

import helpers.constantsNZ as constantsNZ

from utils.BaseDatasetOptions import BaseDatasetOptions

class DatasetOptionsNZ(BaseDatasetOptions):

    def __init__(self, options):
        BaseDatasetOptions.__init__(self);
        self.dir_data = options['dir_data'];
        self.dataset = options['dataset'];

        self.features_categorical = constants.CATEGORICAL_DATA;
        self.features_categorical_binary = None;
        self.data_prefix = 'nz';
        self.subgroups = ['diag'];
        self.featureset = 'newfeatures';
        self.options_featureset = {'names_new_features': new_features};
        self.grouping = 'grouping';
        self.options_grouping = None;
        self.encoding = 'categorical';
        self.options_encoding = None;
        self.options_filtering = None;
        self.chunksize = 10000;
        self.ratio_training_samples = 0.85;

        if options is not None:
            if 'chunksize' in options.keys():
                self.chunksize = options['chunksize'];
            if 'ratio_training_samples' in options.keys():
                self.ratio_training_samples = options['ratio_training_samples'];
            if 'subgroups' in options.keys():
                self.subgroups = options['subgroups'];
            if 'featureset' in options.keys():
                self.featureset = options['featureset'];
            if 'options_featureset' in options.keys():
                self.options_featureset = options['options_featureset'];
            if 'grouping' in options.keys():
                self.grouping = options['grouping'];
            if 'options_grouping' in options.keys():
                self.options_grouping = options['options_grouping'];
            if 'encoding' in options.keys():
                self.encoding = options['encoding'];
            if 'options_encoding' in options.keys():
                self.options_encoding = options['options_encoding'];
            if 'options_filtering' in options.keys():
                self.options_filtering = options['options_filtering'];

        self.filename_options = self._getFilenameOptions(filteroptions=False);
        self.filename = self._getFilename();
        return;


    def getFilenameRawData(self):
        #files are sorted according to years: --> year equals dataset name
        filename_discharge = constants.DISCHARGE_FILE_TEMPLATE.format(self.dataset);
        filename_diagnosis = constants.DIAGNOSIS_FILE_TEMPLATE.format(self.dataset);
        return [filename_discharge, filename_diagnosis];

    def getEventColumnName(self):
        return constants.EVENT_FLAG;

    def getEarlyReadmissionFlagname(self):
        return constants.EARLY_READMISSION_FLAG;

    def getColumnsToRemove(self):
        return constants.COLUMNS_TO_REMOVE_FOR_CLASSIFIER;




