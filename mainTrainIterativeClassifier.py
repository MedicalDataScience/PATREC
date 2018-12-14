import os
import numpy as np
from utils.DatasetOptions import DatasetOptions
from utils.Dataset import Dataset
from utils.Results import Results

from learning.ClassifierRF import ClassifierRF
from learning.ClassifierRF import OptionsRF
from learning.ClassifierLogisticRegression import ClassifierLogisticRegression
from learning.ClassifierLogisticRegression import OptionsLogisticRegression
from learning.ClassifierSGD import ClassifierSGD
from learning.ClassifierSGD import OptionsSGD


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

    dict_opt_sgd = {'loss': 'log', 'penalty': 'l1'};
    options_sgd = OptionsSGD(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_opt_sgd);
    clf_sgd = ClassifierSGD(options_sgd);

    dict_options_dataset_training = {
        'dir_data':         dirData,
        'data_prefix':      'nz',
        'dataset':          '2016',
        'newfeatures':      {'names': constantsNZ.NEW_FEATURES},
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

        results_all_runs_train = Results(dirResultsBase, options_training, options_sgd, 'train');
        results_all_runs_eval = Results(dirResultsBase, options_training, options_sgd, 'eval');

        df_balanced_test = dataset_testing.getBalancedSubSet();

        num_runs = 1;
        eval_aucs = [];
        for run in range(0,num_runs):
            print('');
            [df_balanced_train, df_balanced_eval] = dataset_training.getBalancedSubsetTrainingAndTesting();
            print('train...')
            clf_sgd.train_partial(df_balanced_train, early_readmission_flagname);
            results_train = clf_sgd.predict(df_balanced_train, early_readmission_flagname);
            results_eval = clf_sgd.predict(df_balanced_eval, early_readmission_flagname);
            results_all_runs_train.addResultsSingleRun(results_train);
            results_all_runs_eval.addResultsSingleRun(results_eval);

            auc_train = results_train.getAUC();
            auc_eval = results_eval.getAUC();

            print('train auc: ' + str(auc_train));
            print('eval auc: ' + str(auc_eval));
            eval_aucs.append(auc_eval);
            results_all_runs_train.writeResultsToFileDataset();
            results_all_runs_eval.writeResultsToFileDataset();

        results_all_runs_test = Results(dirResultsBase, options_training, options_sgd, 'test', options_testing);
        early_readmission_flagname = options_testing.getEarlyReadmissionFlagname();
        results_test = clf_sgd.predict(df_balanced_test, early_readmission_flagname);
        results_all_runs_test.addResultsSingleRun(results_test);
        auc_test = results_test.getAUC();
        print('test auc: ' + str(auc_test));

        results_all_runs_test.writeResultsToFileDataset();