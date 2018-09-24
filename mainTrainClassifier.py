import os

from utils.DatasetOptions import DatasetOptions
from utils.CategoricalDataset import CategoricalDataset

from learning.ClassifierRF import ClassifierRF

if __name__ == '__main__':
    dirProject = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
    dirData = dirProject + 'data/';

    new_features = ['previous_visits', 'ratio_los_age', 'ratio_numDK_age', 'ratio_los_numDK', 'ratio_numCHOP_age',
                    'ratio_los_numOE', 'ratio_numOE_age', 'mult_los_numCHOP', 'mult_equalOE_numDK',
                    'diff_drg_alos', 'diff_drg_lowerbound', 'diff_drg_upperbound',
                    'rel_diff_drg_alos', 'rel_diff_drg_lowerbound', 'rel_diff_drg_upperbound',
                    'alos', 'ratio_drg_los_alos'];

    # new_features = ['rel_diff_drg_alos', 'rel_diff_drg_lowerbound', 'rel_diff_drg_upperbound']
    options_standard = None;
    options_newfeatures = {'names_new_features': new_features};
    options_reduction = {'reduction_method': 'NOADMIN'};

    dict_options = {
        'dir_data':                 dirData,
        'dataset':                  '20122015',
        'subgroups':                ['OE', 'DK', 'CHOP'],
        'featureset':               'newfeatures',
        'options_featureset':       options_newfeatures,
        'grouping':                 'grouping',
        'options_grouping':         None,
        'encoding':                 'categorical',
        'options_encoding':         None,
        'options_filtering':        None,
        'chunksize':                10000,
        'ratio_training_samples':   0.85,
    }

    options = DatasetOptions(dict_options)
    dataset_training = CategoricalDataset(dataset_options = options);
    df = dataset_training.getData();

    [df_balanced_training, df_balanced_testing] = dataset_training.getBalancedSubset();

    options_rf = {'n_estimators': 100, 'max_depth': 20, 'n_jobs': 8, 'random_state': None, 'class_weight': None};
    clf_rf = ClassifierRF(options_rf);

    clf_rf.train(df_balanced_training);
    results = clf_rf.predict(df_balanced_testing);
    auc = results.getAUC();
    print('auc: ' + str(auc))

