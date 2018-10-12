from preprocessing.ColumnSplitter import ColumnSplitter
from preprocessing.DataCleaner import DataCleaner
from preprocessing.DataPreparer import DataPreparer
from preprocessing.DataGrouper import DataGrouper
from preprocessing.FeatureEncoder import FeatureEncoder
from preprocessing.FeatureSetCreator import FeatureSetCreator


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

    def __init__(self, options_dataset):
        self.options = options_dataset;
        return;


    def splitColumns(self):
        splitter = ColumnSplitter(self.options);
        splitter.splitColumnsAllSubgroups();


    def clean(self):
        filename_options_in = None
        filename_options_out = 'clean'
        cleaner = DataCleaner(self.options, filename_options_in, filename_options_out)
        cleaner.cleanData()


    def group(self):
        grouper = DataGrouper(self.options);
        grouper.groupFeatures()


    def createFeatureSet(self):
        featureset_creator = FeatureSetCreator(self.options);
        featureset_creator.createFeatureSet();


    def encodeFeatures(self):
        encoder = FeatureEncoder(self.options);
        encoder.encodeFeatures();


    def fuse(self):
        preparer = DataPreparer(self.options);
        preparer.fuseSubgroups();








