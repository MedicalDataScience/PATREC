
import sys
import os

from shutil import copy2

from preprocessing.FeatureCreator import FeatureCreator
from preprocessing.FeatureReducer import FeatureReducer


class FeatureSetCreator:
    # create the feature set of the REST columns
    def __init__(self, options_dataset, filename_options_in='clean'):
        self.options = options_dataset;
        self.filename_options_in = filename_options_in;
        return;

    def createFeatureSet(self):
        dir_data = self.options.getDirData();
        data_prefix = self.options.getDataPrefix();
        dataset = self.options.getDatasetName();
        subgroups = self.options.getSubgroups();
        name_dem_features = self.options.getFilenameOptionDemographicFeatures();
        newfeatures = self.options.getNewFeatureSettings();
        featurereduction = self.options.getFeatureReductionSettings();

        if newfeatures is None and featurereduction is None:
            filename_data_in = os.path.join(dir_data, data_prefix + '_' + dataset + '_' + name_dem_features + '_' + self.filename_options_in + '.csv');
            filename_data_out = os.path.join(dir_data, data_prefix + '_' + dataset + '_' + name_dem_features + '_standard.csv');
            copy2(filename_data_in, filename_data_out);
        else:
            if newfeatures is not None:
                #assert subgroups is not None, 'subgroups are needed to create new features...exit'
                filename_options_out = 'newfeatures';
                creator = FeatureCreator(self.options, self.filename_options_in, filename_options_out)
                creator.addFeatures();
            if featurereduction is not None:
            # reduce features
                #assert featureset_options['reduction_method'] is not None, 'there has to be a feature reduction method for this choice of feature set...exit';
                if newfeatures is not None:
                    filename_options_in = 'newfeatures';
                else:
                    filename_options_in = self.filename_options_in;
                reducer = FeatureReducer(self.options, filename_options_in);
                reducer.reduceFeatures();
        # else:
        #     print('name of feature set is not known: ' + str(name))
        #     print('..exit!');
        #     sys.exit();