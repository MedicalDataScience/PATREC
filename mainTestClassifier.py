import os
import numpy as np

from utils.DatasetOptions import DatasetOptions
from utils.Dataset import Dataset
from utils.Results import Results

from learning.ClassifierRF import ClassifierRF
from learning.ClassifierRF import OptionsRF
from learning.ClassifierLogisticRegression import ClassifierLogisticRegression
from learning.ClassifierLogisticRegression import OptionsLogisticRegression

import helpers.constants as constantsPATREC

if __name__ == '__main__':
    dirProject = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
    dirData = dirProject + 'data/';
    dirResultsBase = dirProject + 'results/';
    dirModelsBase = dirProject + 'classifiers/'

    dict_options_dataset_training = {
        'dir_data':                 dirData,
        'data_prefix':              'patrec',
        'dataset':                  '20122015',
        'encoding':                 'categorical',
        'newfeatures':              {'names': constantsPATREC.NEW_FEATURES},
        'featurereduction':         None,
        'grouping':                 'verylightgrouping',
        'filtering':                'oncology'
    }
    dict_options_dataset_testing = {
        'dir_data':                 dirData,
        'data_prefix':              'patrec',
        'dataset':                  '20162017',
        'encoding':                 'categorical',
        'newfeatures':              {'names': constantsPATREC.NEW_FEATURES},
        'featurereduction':         None,
        'grouping':                 'verylightgrouping',
        'filtering':                'oncology'
    }

    options_training = DatasetOptions(dict_options_dataset_training);
    dataset_training = Dataset(dataset_options=options_training);

    dict_opt_rf = {'n_estimators': 100, 'max_depth': 15};
    options_rf = OptionsRF(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_opt_rf);
    clf_rf = ClassifierRF(options_rf);

    dict_opt_lr = {'penalty': 'l1', 'C': 0.1};
    options_lr = OptionsLogisticRegression(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_opt_lr);
    clf_lr = ClassifierLogisticRegression(options_lr);

    options_clf = options_rf
    clf = clf_rf;

    options_testing = DatasetOptions(dict_options_dataset_testing);
    dataset_testing = Dataset(dataset_options=options_testing);
    results_all_runs_test = Results(dirResultsBase, options_training, options_clf, 'test', options_testing);

    early_readmission_flagname = options_testing.getEarlyReadmissionFlagname();

    test_aucs = [];
    num_runs = 10;
    for k in range(0, num_runs):
        df_balanced_test = dataset_testing.getBalancedSubSet();
        clf.loadFromFile(k);
        results_test = clf.predict(df_balanced_test, early_readmission_flagname);
        auc_test = results_test.getAUC();
        test_aucs.append(auc_test);
        print('test auc: ' + str(auc_test));
        results_all_runs_test.addResultsSingleRun(results_test);

    results_all_runs_test.writeResultsToFileDataset();
    print('')
    print('mean test auc: ' + str(np.mean(np.array(test_aucs))))
    print('')
