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
import helpers.constantsNZ as constantsNZ

if __name__ == '__main__':
    dirProject = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
    dirData = dirProject + 'data/';
    dirResultsBase = dirProject + 'results/';
    dirModelsBase = dirProject + 'classifiers/'

    dict_options_dataset_training = {
        'dir_data':         dirData,
        'data_prefix':      'nz',
        'dataset':          str(2012),
        'newfeatures':      {'names': constantsNZ.NEW_FEATURES},
        'featurereduction': None
    }
    options_training = DatasetOptions(dict_options_dataset_training);

    dict_opt_rf = {'n_estimators': 100, 'max_depth': 15, 'warm_start': True};
    options_rf = OptionsRF(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_opt_rf);
    clf_rf = ClassifierRF(options_rf);

    dict_options_dataset_training = {
        'dir_data':         dirData,
        'data_prefix':      'nz',
        'dataset':          '2016',
        'newfeatures':              {'names': constantsNZ.NEW_FEATURES},
        'featurereduction': None
    }
    options_testing = DatasetOptions(dict_options_dataset_training);
    dataset_testing = Dataset(dataset_options=options_testing);

    years = [2012, 2013, 2014, 2015];
    for year in years:
        dict_options_dataset_training = {
            'dir_data':                 dirData,
            'data_prefix':              'nz',
            'dataset':                  str(year),
            'newfeatures':              {'names': constantsNZ.NEW_FEATURES},
            'featurereduction':         None
        }

        options_training = DatasetOptions(dict_options_dataset_training);
        dataset_training = Dataset(dataset_options=options_training);
        early_readmission_flagname = options_training.getEarlyReadmissionFlagname();

        print('dataset filename: ' + str(dataset_training.getFilename()))

        results_all_runs_train = Results(dirResultsBase, options_training, options_rf, 'train');
        results_all_runs_eval = Results(dirResultsBase, options_training, options_rf, 'eval');

        num_runs = 1;
        eval_aucs = [];
        for run in range(0,num_runs):
            print('');
            [df_balanced_train, df_balanced_eval] = dataset_training.getBalancedSubsetTrainingAndTesting();

            clf_rf.train(df_balanced_train, early_readmission_flagname);
            results_train = clf_rf.predict(df_balanced_train, early_readmission_flagname);
            results_eval = clf_rf.predict(df_balanced_eval, early_readmission_flagname);
            results_all_runs_train.addResultsSingleRun(results_train);
            results_all_runs_eval.addResultsSingleRun(results_eval);

            auc_train = results_train.getAUC();
            auc_eval = results_eval.getAUC();

            print('train auc: ' + str(auc_train));
            print('eval auc: ' + str(auc_eval));
            # clf.save(run);
            # clf.saveLearnedFeatures(run)

            eval_aucs.append(auc_eval);

        # results_all_runs_train.writeResultsToFileDataset();
        # results_all_runs_eval.writeResultsToFileDataset();
        # print('')
        # print('mean eval auc: ' + str(np.mean(np.array(eval_aucs))))
        # print('')

    early_readmission_flagname = options_testing.getEarlyReadmissionFlagname();
    df_balanced_test = dataset_testing.getBalancedSubSet();
    results_test = clf_rf.predict(df_balanced_test, early_readmission_flagname);
    auc_test = results_test.getAUC();
    print('test auc: ' + str(auc_test));