
import sys
import os

from shutil import copy2

from preprocessing.FeatureCreator import FeatureCreator
from preprocessing.FeatureReducer import FeatureReducer


class FeatureSetCreator:
    # create the feature set of the REST columns
    def __init__(self, options_dataset):
        self.options = options_dataset;
        self.filename_options_in = 'clean';
        return;

    def createFeatureSet(self):
        dir_data = self.options.getDirData();
        dataset = self.options.getDatasetName();
        name = self.options.getFeatureSet();
        featureset_options = self.options.getFeatureSetOptions();
        subgroups = self.options.getSubgroups();

        if name == 'standard':
            filename_data_in = dir_data + 'data_' + dataset + '_REST_' + self.filename_options_in + '.csv';
            filename_data_out = dir_data + 'data_' + dataset + '_REST_' + name + '.csv';
            copy2(filename_data_in, filename_data_out);
        elif name == 'newfeatures':
            assert subgroups is not None, 'subgroups are needed to create new features...exit'
            filename_options_out = 'newfeatures';
            creator = FeatureCreator(self.options, self.filename_options_in, filename_options_out)
            creator.addFeatures();
        elif name == 'reduction':
        # reduce features
            assert featureset_options['reduction_method'] is not None, 'there has to be a feature reduction method for this choice of feature set...exit';
            reducer = FeatureReducer(self.options, self.filename_options_in);
            reducer.reduceFeatures();
        else:
            print('name of feature set is not known: ' + str(name))
            print('..exit!');
            sys.exit();