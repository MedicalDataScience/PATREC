import os
import sys
from preprocessing.ColumnSplitter import ColumnSplitter
from preprocessing.DataCleaner import DataCleaner
from preprocessing.DataPreparer import DataPreparer
from preprocessing.DataGrouper import DataGrouper
from preprocessing.DataSplitter import DataSplitter
from preprocessing.FeatureEncoder import FeatureEncoder
from preprocessing.FeatureSetCreator import FeatureSetCreator
from preprocessing.DataFilter import DataFilter


# Still Missing/ TODO:
#   DataGrouper:
#   - the possibility to include count variables in the grouping functions (instead of only binary variables)
#       the effect is unclear though; therefore wait with implementing until its proven to be worse if we only use
#       binary variables for CHOP_xx, DK_x, etc
#   - there is only a single grouping functionality implemented atm; although it can be easily changed in the code
#       the argument option has still to be implemented. There was not a big impact on the performance, hence there was no
#       priority for implementing that
#   Filtering based on values of a feature
#   - mostly for testing it is useful to be able to filter data based on values in the columns, e.g. EntlassBereich.
#       this functionality is not yet implemented in the class structure
#
#   ...probably there is more to come of what is not yet implemented/missing

class Preprocessor():

    def __init__(self, options_preprocessing):
        self.filename_data_1 = '/scicore/home/vogtju/GROUP/PATREC_USB/PR15_2012-15___Anonym18Konsent.csv';
        self.filename_data_2 = '/scicore/home/vogtju/GROUP/PATREC_USB/PR15_2016-17___Anonym18Konsent.csv';
        self.DIR_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/';
        self.dir_data = self.DIR_PROJECT + 'data/';

        self.dataset = options_preprocessing['dataset'];
        self.chunksize = options_preprocessing['chunksize'];
        self.subgroup_names = options_preprocessing['subgroups'];

        self.featureset = options_preprocessing['featureset'];
        self.featureset_options = options_preprocessing['options_featureset'];
        self.encoding = options_preprocessing['encoding'];
        self.grouping = options_preprocessing['grouping'];

        if self.dataset == '20122015':
            self.filename_raw = self.filename_data_1;
        elif self.dataset == '20162017':
            self.filename_raw = self.filename_data_2;
        else:
            print('dataset is unknown... exit')
            sys.exit();
        return;


    def splitColumns(self):
        splitter = ColumnSplitter(self.filename_raw, self.chunksize);
        for g in self.subgroup_names:
            filename_out = self.dir_data + 'data_' + self.dataset + '_' + g + '.csv';
            splitter.splitColumns(g, filename_out);

        filename_out_rest = self.dir_data + 'data_' + self.dataset + '_REST.csv';
        splitter.splitColumns('REST', filename_out_rest);


    def clean(self, filename_options_in=None, filename_options_out='clean'):
        cleaner = DataCleaner(self.dir_data, self.dataset, filename_options_in, filename_options_out)
        cleaner.cleanData(self.subgroup_names)


    def group(self):
        grouper = DataGrouper(self.dir_data, self.dataset, self.grouping, self.subgroup_names);
        grouper.groupFeatures()


    def createFeatureSet(self):
        featureset_creator = FeatureSetCreator(self.dir_data, self.dataset);
        featureset_creator.createFeatureSet(self.featureset, self.featureset_options);


    def encodeFeatures(self):
        filename_options_in = self.featureset;
        encoder = FeatureEncoder(self.dir_data, self.dataset, filename_options_in);
        encoder.encodeFeatures(self.encoding);


    def fuse(self):
        filename_options_in = self.featureset + '_' + self.encoding;
        filename_options_out = self.featureset + '_' + self.encoding + '_' + self.grouping;
        preparer = DataPreparer(self.dir_data, self.dataset, filename_options_in, filename_options_out, self.grouping);
        preparer.fuseSubgroups(self.subgroup_names, self.encoding, self.featureset, self.featureset_options);


    def filterData(self, filterKey):
        filename_options_in = self.featureset + '_' + self.encoding + '_' + self.grouping;
        filter = DataFilter(self.dir_data, self.dataset, filename_options_in)
        filter.filterDataBinaryColumns(filterKey)


    def splitDatasetIntoTrainingTestingSet(self):
        filename_options_in = self.featureset + '_' + self.encoding + '_' + self.grouping;
        datasplitter = DataSplitter(self.dir_data, self.dataset, filename_options_in)
        datasplitter.splitDatasetIntoTrainingTesting();








