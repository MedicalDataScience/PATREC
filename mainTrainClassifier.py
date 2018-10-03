import os
import numpy as np
from utils.DatasetOptions import DatasetOptions
from utils.CategoricalDataset import CategoricalDataset
from utils.Results import Results

from learning.ClassifierRF import ClassifierRF
from learning.ClassifierRF import OptionsRF

if __name__ == '__main__':
    dirProject = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
    dirData = dirProject + 'data/';
    dirResultsBase = dirProject + 'results/';
    dirModelsBase = dirProject + 'classifiers/'

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
    dict_options_dataset_testing = {
        'dir_data': dirData,
        'dataset': '20162017',
        'subgroups': ['OE', 'DK', 'CHOP'],
        'featureset': 'newfeatures',
        'options_featureset': options_newfeatures,
        'grouping': 'grouping',
        'options_grouping': None,
        'encoding': 'categorical',
        'options_encoding': None,
        'options_filtering': None,
        'chunksize': 10000,
        'ratio_training_samples': 0.85,
    }


    options_training = DatasetOptions(dict_options_dataset_training);
    dataset_training = CategoricalDataset(dataset_options=options_training);
    options_testing = DatasetOptions(dict_options_dataset_testing);
    dataset_testing = CategoricalDataset(dataset_options=options_testing);
    options_rf = OptionsRF(dirModelsBase, options_training.getFilenameOptions());
    clf_rf = ClassifierRF(options_rf);

    results_all_runs_train = Results(dirResultsBase, options_training, options_testing, options_rf, 'train');
    results_all_runs_eval = Results(dirResultsBase, options_training, options_testing, options_rf, 'eval');
    num_runs = 10;
    eval_aucs = [];
    for run in range(0,num_runs):
        [df_balanced_train, df_balanced_eval] = dataset_training.getBalancedSubsetTrainingAndTesting();
        # df_balanced_test = dataset_testing.getBalancedSubSet();

        clf_rf.train(df_balanced_train);
        results_train = clf_rf.predict(df_balanced_train);
        results_eval = clf_rf.predict(df_balanced_eval);
        # results_test = clf_rf.predict(df_balanced_test);

        auc_train = results_train.getAUC();
        auc_eval = results_eval.getAUC();
        # auc_test = results_test.getAUC();

        print('');
        print('train auc: ' + str(auc_train));
        print('eval auc: ' + str(auc_eval));
        # print('test auc: ' + str(auc_test));
        clf_rf.save(run);
        clf_rf.saveLearnedFeatures(run)

        eval_aucs.append(auc_eval);

    results_all_runs_train.writeResultsToFileDataset();
    results_all_runs_eval.writeResultsToFileDataset();

    print('mean eval auc: ' + str(np.mean(np.array(eval_aucs))))
