import os
import numpy as np
from utils.DatasetOptions import DatasetOptions
from utils.Dataset import Dataset
from utils.Results import Results

from learning.ClassifierRF import ClassifierRF
from learning.ClassifierRF import OptionsRF
from learning.ClassifierLogisticRegression import ClassifierLogisticRegression
from learning.ClassifierLogisticRegression import OptionsLogisticRegression
from learning.ClassifierSVM import OptionsSVM
from learning.ClassifierSVM import ClassifierSVM

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
    'grouping':                 'verylightgrouping',
    'encoding':                 'categorical',
    'newfeatures':              {'names': constantsPATREC.NEW_FEATURES},
    'featurereduction':         None,
    'filtering':                'oncology'
}

    options_training = DatasetOptions(dict_options_dataset_training);
    dataset_training = Dataset(dataset_options=options_training);
    early_readmission_flagname = options_training.getEarlyReadmissionFlagname();

    print('dataset filename: ' + str(dataset_training.getFilename()))

    dict_opt_rf = {'n_estimators': 100, 'max_depth': 5};
    options_rf = OptionsRF(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_opt_rf);
    clf_rf = ClassifierRF(options_rf);

    dict_opt_lr = {'penalty': 'l2', 'C': 0.0001};
    options_lr = OptionsLogisticRegression(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_opt_lr);
    clf_lr = ClassifierLogisticRegression(options_lr);


    dict_options_svm = {'kernel': 'rbf', 'C': 1.0};
    options_svm = OptionsSVM(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_options_svm)
    clf_svm = ClassifierSVM(options_svm);

    options_clf = options_rf
    clf = clf_rf;

    results_all_runs_train = Results(dirResultsBase, options_training, options_clf, 'train');
    results_all_runs_eval = Results(dirResultsBase, options_training, options_clf, 'eval');

    num_runs = 10;
    eval_aucs = [];
    for run in range(0,num_runs):
        print('');
        [df_balanced_train, df_balanced_eval] = dataset_training.getBalancedSubsetTrainingAndTesting();

        clf.train(df_balanced_train, early_readmission_flagname);
        results_train = clf.predict(df_balanced_train, early_readmission_flagname);
        results_eval = clf.predict(df_balanced_eval, early_readmission_flagname);
        results_all_runs_train.addResultsSingleRun(results_train);
        results_all_runs_eval.addResultsSingleRun(results_eval);

        auc_train = results_train.getAUC();
        auc_eval = results_eval.getAUC();

        print('train auc: ' + str(auc_train));
        print('eval auc: ' + str(auc_eval));
        clf.save(run);
        clf.saveLearnedFeatures(run)

        eval_aucs.append(auc_eval);

    results_all_runs_train.writeResultsToFileDataset();
    results_all_runs_eval.writeResultsToFileDataset();
    print('')
    print('mean eval auc: ' + str(np.mean(np.array(eval_aucs))))
    print('')

