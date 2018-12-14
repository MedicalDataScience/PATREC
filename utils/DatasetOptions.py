import sys

from utils.BaseDatasetOptions import BaseDatasetOptions

import helpers.constants as constantsPATREC
import helpers.constantsNZ as constantsNZ

import helpers.helpers as helpers
import helpers.icd10_chapters as icd10_chapters

class DatasetOptions(BaseDatasetOptions):

    def __init__(self, options):
        BaseDatasetOptions.__init__(self)
        self.data_prefix = options['data_prefix'];
        self.dir_data = options['dir_data'];
        self.dataset = options['dataset'];

        self.features_categorical = self._getCategoricalFeatures();
        self.subgroups = self._getSubgroups();
        self.newfeatures = None;
        self.featurereduction = None;
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
            if 'newfeatures' in options.keys():
                self.newfeatures = options['newfeatures'];              #self.newfeatures = {'names': self._getNamesNewFeatures()};
            if 'featurereduction' in options.keys():
                self.featurereduction = options['featurereduction'];    #self.reduction = {'method': self._getNamesNewFeatures()}
            if 'grouping' in options.keys():
                self.grouping = options['grouping'];
            if 'options_grouping' in options.keys():
                self.options_grouping = options['options_grouping'];
            if 'encoding' in options.keys():
                self.encoding = options['encoding'];
            if 'options_encoding' in options.keys():
                self.options_encoding = options['options_encoding'];
            if 'filtering' in options.keys():
                self.options_filtering = options['filtering'];

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


    def getNameMainDiag(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.HAUPTDIAGNOSE;
        elif self.data_prefix == 'nz':
            return constantsNZ.HAUPTDIAGNOSE;
        else:
            print('data prefix is unknown...exit')
            sys.exit();

    def getNameSecDiag(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.NEBENDIAGNOSE;
        elif self.data_prefix == 'nz':
            return constantsNZ.NEBENDIAGNOSE;
        else:
            print('data prefix is unknown...exit')
            sys.exit();

    def getAdminFeatureNames(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.ADMIN_FEATURES_NAMES.copy();
        elif self.data_prefix == 'nz':
            return constantsNZ.ADMIN_FEATURES_NAMES.copy();


    def getLiegestatusFeatureNames(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.LIEGESTATUS_FEATURES.copy();
        else:
            print('no LIEGESTATUS defined for this dataset...')
            return [];

    def getFilenameOptionDemographicFeatures(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.NAME_DEMOGRAPHIC_FEATURE;
        elif self.data_prefix == 'nz':
            return constantsNZ.NAME_DEMOGRAPHIC_FEATURE;
        else:
            print('there does not exist yet any demographic information for this dataset..exit')
            sys.exit();

    def getFusionFeatureNames(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.FUSION_FEATURES.copy();
        elif self.data_prefix == 'nz':
            return constantsNZ.FUSION_FEATURES.copy();
        else:
            print('no FUSION features for this dataset yet...')
            sys.exit();


    def getDiagGroupNames(self):
        if self.grouping == 'verylightgrouping':
            group_names = helpers.getDKverylightGrouping();
        elif self.grouping == 'lightgrouping':
            group_names = helpers.getDKlightGrouping();
        elif self.grouping == 'grouping':
            group_names = helpers.getDKgrouping();
        else:
            group_names = [];
        return group_names;


    def getColumnsToRemove(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.COLUMNS_TO_REMOVE_FOR_CLASSIFIER;
        elif self.data_prefix == 'nz':
            return constantsNZ.COLUMNS_TO_REMOVE_FOR_CLASSIFIER;


    def getDiseaseICDkeys(self):
        if self.options_filtering == 'cardiovascular':
            keys = icd10_chapters.getCodesMainGroup('I00-I99');
        elif self.options_filtering == 'oncology':
            keys1 = icd10_chapters.getCodesSubgroup('C00-D48', 'C00-C97');
            keys2 = icd10_chapters.getCodesSubgroup('C00-D48', 'D00-D09');
            keys = keys1+keys2;
        elif self.options_filtering == 'chronic_lung':
            keys1 = icd10_chapters.getCodesSubgroup('J00-J99', 'J30-J39')
            keys2 = icd10_chapters.getCodesSubgroup('J00-J99', 'J40-J47')
            keys3 = icd10_chapters.getCodesSubgroup('J00-J99', 'J60-J70')
            keys4 = icd10_chapters.getCodesSubgroup('J00-J99', 'J80-J84')
            keys5 = icd10_chapters.getCodesSubgroup('J00-J99', 'J85-J86')
            keys6 = icd10_chapters.getCodesSubgroup('J00-J99', 'J90-J94')
            keys7 = icd10_chapters.getCodesSubgroup('J00-J99', 'J95-J99')
            keys = keys1+keys2+keys3+keys4+keys5+keys6+keys7;
        else:
            print('unknown disease...exit')
            sys.exit()
        return keys;


    def getDiseaseNames(self):
        if self.data_prefix == 'patrec':
            return constantsPATREC.DISEASES.copy();
        elif self.data_prefix == 'nz':
            return constantsNZ.DISEASES.copy();
        else:
            print('unknown dataset...exit')
            sys.exit();


    # def getLOSState(self):
    #     return constants.getLOSState();
    #
    # def getAgeState(self):
    #     return constants.getAgeState();
    #
    # def getCountFeaturesToBinarize(self):
    #     return constants.getCountFeaturesToBinarize();
