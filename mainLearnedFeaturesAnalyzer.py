
import os

import matplotlib.pyplot as plt

from utils.DatasetOptions import DatasetOptions
from utils.Dataset import Dataset
from learning.ClassifierRF import OptionsRF
from learning.ClassifierLogisticRegression import OptionsLogisticRegression
from analyzing.LearnedFeaturesAnalyzer import LearnedFeaturesAnalyzer

import helpers.constants as constantsPATREC
import helpers.constantsNZ as constantsNZ

def getMostImportantFeaturesSingleRun(feature_values, feature_names, num_features):
    feat_coefs = [];
    for k in range(0, len(feature_values)):
        feat_name = feature_names[k];
        feat_value = feature_values[k];
        feat_coefs.append([abs(feat_value), feat_name]);

    sorted_importances = sorted(feat_coefs, key=lambda x: x[0], reverse=True);
    most_important_features = sorted_importances[:num_features];
    return most_important_features;


if __name__ == '__main__':
    dirProject = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
    dirData = dirProject + 'data/';
    dirResultsBase = dirProject + 'results/';
    dirModelsBase = dirProject + 'classifiers/'
    dirPlotsBase = dirProject + 'plots/learned_features/';

    dict_options_dataset_training = {
        'dir_data':             dirData,
        'data_prefix':          'patrec',
        'dataset':              '20122015',
        'subgroups':            ['DK'],
        'encoding':             'categorical',
        'newfeatures':          None,
        'featurereduction':     {'method': 'FUSION'},
        'grouping':             'verylightgrouping'
    }
    options_training = DatasetOptions(dict_options_dataset_training);

    dict_opt_rf = {'n_estimators': 100, 'max_depth': 15};
    options_rf = OptionsRF(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_opt_rf);

    dict_opt_lr = {'penalty': 'l2', 'C': 0.01};
    options_lr = OptionsLogisticRegression(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_opt_lr);

    options = options_rf;

    analyzer = LearnedFeaturesAnalyzer(dir_plots_base=dirPlotsBase,
                                       options_training_data=options_training,
                                       options_classifier=options,
                                       num_runs=10);

    num_important_features = 25;
    analyzer.plotAvgLearnedFeatures(num_features=num_important_features);








