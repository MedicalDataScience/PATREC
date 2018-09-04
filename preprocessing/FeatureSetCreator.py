
import sys
import os

from shutil import copy2

from preprocessing.FeatureCreator import FeatureCreator
from preprocessing.FeatureReducer import FeatureReducer


class FeatureSetCreator:
    # create the feature set of the REST columns
    def __init__(self, dir_data, dataset):
        self.dir_data = dir_data;
        self.dataset = dataset;
        self.filename_options_in = 'clean';
        return;

    def createFeatureSet(self, name, options):

        if name == 'standard':
            filename_data_in = self.dir_data + 'data_' + self.dataset + '_REST_' + self.filename_options_in + '.csv';
            filename_data_out = self.dir_data + 'data_' + self.dataset + '_REST_' + name + '.csv';
            copy2(filename_data_in, filename_data_out);
        elif name == 'newfeatures':
            assert options['subgroups'] is not None, 'subgroups are needed to create new features...exit'
            assert options['names_new_features'] is not None, 'names new features cannot be None when choosing this featureset...exit'

            subgroups = options['subgroups']
            names_new_features = options['names_new_features'];

            if len(names_new_features) == 0:
                print('WARNING: empty list of names for new features...');
            filename_options_out = 'newfeatures';
            creator = FeatureCreator(self.dir_data, self.dataset, self.filename_options_in, filename_options_out, subgroups, names_new_features)
            creator.addFeatures();
        # add features
        # fuse columns
        elif name == 'reduction':
        # reduce features
            assert options['reduction_method'] is not None, 'there has to be a feature reduction method for this choice of feature set...exit';
            reduction_method = options['reduction_method'];
            filename_options_out = 'reduction' + reduction_method;
            reducer = FeatureReducer(self.dir_data, self.dataset, self.filename_options_in, filename_options_out);
            reducer.reduceFeatures(reduction_method);
        else:
            print('name of feature set is not known: ' + str(name))
            print('..exit!');
            sys.exit();