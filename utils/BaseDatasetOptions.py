
import sys

class BaseDatasetOptions:

    def __init__(self):
        self.dir_data = None;
        self.dataset = None;
        self.data_prefix = None;

        self.features_categorical = None;

        self.subgroups = None;
        self.newfeatures = None;
        self.featurereduction = None;
        self.grouping = None;
        self.options_grouping = None;
        self.encoding = None;
        self.options_encoding = None;
        self.options_filtering = None;
        self.chunksize = None;
        self.ratio_training_samples = None;

        self.filename_options = None;
        self.filename = None;
        return;


    def _getFilenameOptions(self, filteroptions):
        str = self.data_prefix + '_' + self.dataset;
        str = str + '_' + self.getFeatureSetStr()

        str = str + '_' + self.encoding;
        if self.options_encoding is not None:
            str = str + self.options_grouping;

        str = str + '_' + self.grouping;
        if self.options_grouping is not None:
            str = str + self.options_grouping;

        if filteroptions:
            if self.options_filtering is not None:
                str = str + '_' + self.options_filtering;
        self.filename_options = str;


    def _getFilename(self):
        str = self.dir_data + '/' + 'data_';
        if self.filename_options is None:
            self._getFilenameOptions(filteroptions=False);
        str = str + self.filename_options;
        str = str + '.csv';
        return str;


    def getDirData(self):
        return self.dir_data;

    def getDataPrefix(self):
        return self.data_prefix;

    def getDatasetName(self):
        return self.dataset;


    def getSubgroups(self):
        return self.subgroups;


    def getGroupingName(self):
        return self.grouping;


    def getEncodingScheme(self):
        return self.encoding;


    def getChunkSize(self):
        return self.chunksize;


    def getRatioTrainingSamples(self):
        return self.ratio_training_samples;


    def getOptionsFiltering(self):
        return self.options_filtering;


    def getFilenameOptions(self, filteroptions):
        self._getFilenameOptions(filteroptions);
        return self.filename_options;


    def getFilename(self):
        if self.filename is None:
            self.filename = self._getFilename();
        return self.filename;


    def getFilenameSubgroup(self, subgroup):
        str = self.dir_data + '/' + 'data_';
        str = str + self.data_prefix + '_' + self.dataset + '_' + subgroup + '_' + self.grouping + '.csv';
        return str;


    def getNewFeatureSettings(self):
        return self.newfeatures;

    def getFeatureReductionSettings(self):
        return self.featurereduction;

    def getFeatureSetStr(self):
        if self.newfeatures is None and self.featurereduction is None:
            str = 'standard';
        else:
            if self.newfeatures is not None and self.featurereduction is not None:
                str = 'newfeatures_reduction_' + self.featurereduction['method'];
            elif self.newfeatures is not None:
                str = 'newfeatures';
            elif self.featurereduction is not None:
                str = 'reduction_' + self.featurereduction['method'];
            else:
                print('this should not happen...exit')
                sys.exit();
        return str;


    def getCategoricalFeatures(self):
        return list(self.features_categorical.keys());

    def getFeatureCategories(self, feature):
        try:
            return list(self.features_categorical[feature]).copy();
        except ValueError:
            print('Categorical feature does not exist for this dataset...exit')
            sys.exit();
