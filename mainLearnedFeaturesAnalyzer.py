
import os

import matplotlib.pyplot as plt

from utils.DatasetOptions import DatasetOptions
from utils.CategoricalDataset import CategoricalDataset
from learning.ClassifierRF import OptionsRF
from analyzing.LearnedFeaturesAnalyzer import LearnedFeaturesAnalyzer



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

    new_features = ['previous_visits', 'ratio_los_age', 'ratio_numDK_age', 'ratio_los_numDK', 'ratio_numCHOP_age',
                    'ratio_los_numOE', 'ratio_numOE_age', 'mult_los_numCHOP', 'mult_equalOE_numDK',
                    'diff_drg_alos', 'diff_drg_lowerbound', 'diff_drg_upperbound',
                    'rel_diff_drg_alos', 'rel_diff_drg_lowerbound', 'rel_diff_drg_upperbound',
                    'alos', 'ratio_drg_los_alos'];

    # new_features = ['rel_diff_drg_alos', 'rel_diff_drg_lowerbound', 'rel_diff_drg_upperbound']
    options_standard = None;
    options_newfeatures = {'names_new_features': new_features};
    options_reduction = {'reduction_method': 'NOADMIN'};

    dict_options_dataset_training = {
        'dir_data':                 dirData,
        'dataset':                  '20122015',
        'subgroups':                ['OE', 'DK', 'CHOP'],
        'featureset':               'newfeatures',
        'options_featureset':       options_newfeatures,
        'grouping':                 'grouping',
        'options_grouping':         None,
        'encoding':                 'categorical',
        'options_encoding':         None,
        'options_filtering':        'EntlassBereich_Med',
        'chunksize':                10000,
        'ratio_training_samples':   0.85,
    }


    options_training = DatasetOptions(dict_options_dataset_training);
    dataset_training = CategoricalDataset(dataset_options=options_training);
    options_rf = OptionsRF(dirModelsBase, options_training.getFilenameOptions());

    analyzer = LearnedFeaturesAnalyzer(dir_plots_base=dirPlotsBase,
                                       options_training_data=options_training,
                                       options_classifier=options_rf,
                                       dataset_training=dataset_training,
                                       num_runs=10);

    num_important_features = 50;
    analyzer.plotAvgLearnedFeatures(num_features=num_important_features);








