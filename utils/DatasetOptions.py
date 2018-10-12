import sys

from utils.BaseDatasetOptions import BaseDatasetOptions

import helpers.constants as constantsPATREC
import helpers.constantsNZ as constantsNZ

class DatasetOptions(BaseDatasetOptions):

    def __init__(self, options):
        BaseDatasetOptions.__init__(self)
        self.data_prefix = options['data_prefix'];
        self.dir_data = options['dir_data'];
        self.dataset = options['dataset'];

        self.features_categorical = self._getCategoricalFeatures();
        self.subgroups = self._getSubgroups();
        self.featureset = 'newfeatures';
        self.options_featureset = {'names_new_features': self._getNamesNewFeatures()};
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
            if 'subgroups'  in options.keys():
                self.subgroups = options['subgroups'];
            if 'featureset' in options.keys():
                self.featureset = options['featureset'];
            if 'options_featureset'  in options.keys():
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


    def _getCategoricalFeatures(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.CATEGORICAL;
        elif self.data_prefix == 'nz':
            return constantsNZ.CATEGORICAL_DATA;


    def _getSubgroups(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.SUBGROUPS;
        elif self.data_prefix == 'nz':
            return ['diag'];


    def _getNamesNewFeatures(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.NEW_FEATURES;
        elif self.data_prefix == 'nz':
            return constantsNZ.NEW_FEATURES;


    def getFilenameRawData(self):
        if self.data_prefix == 'patrec':
            if self.dataset == '20122015':
                filename = constantsPATREC.filename_data_20122015;
            elif self.dataset == '20122015':
                filename = constantsPATREC.filename_data_20162017;
            else:
                print('dataset is not known...exit')
                sys.exit()
            return filename;
        elif self.data_prefix == 'nz':
            # files are sorted according to years: --> year equals dataset name
            filename_discharge = constantsNZ.DISCHARGE_FILE_TEMPLATE.format(self.dataset);
            filename_diagnosis = constantsNZ.DIAGNOSIS_FILE_TEMPLATE.format(self.dataset);
            return [filename_discharge, filename_diagnosis];


    def getEventColumnName(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.EVENT_FLAG;
        elif self.data_prefix == 'nz':
            return constantsNZ.EVENT_FLAG;


    def getEarlyReadmissionFlagname(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.EARLY_READMISSION_FLAG;
        elif self.data_prefix == 'nz':
            return constantsNZ.EARLY_READMISSION_FLAG;

    def getAdminFeatureNames(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.ADMIN_FEATURES_NAMES.copy();
        elif self.data_prefix == 'nz':
            return constantsNZ.ADMIN_FEATURES_NAMES.copy();

    def getColumnsToRemove(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.COLUMNS_TO_REMOVE_FOR_CLASSIFIER;
        elif self.data_prefix == 'nz':
            return constantsNZ.COLUMNS_TO_REMOVE_FOR_CLASSIFIER;




    # def getLOSState(self):
    #     return constants.getLOSState();
    #
    # def getAgeState(self):
    #     return constants.getAgeState();
    #
    # def getCountFeaturesToBinarize(self):
    #     return constants.getCountFeaturesToBinarize();
