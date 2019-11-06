import os
import numpy as np
from utils.DatasetOptions import DatasetOptions
from utils.Dataset import Dataset
from utils.Results import Results

from learning.ClassifierRF import ClassifierRF
from learning.ClassifierRF import OptionsRF
from learning.ClassifierLogisticRegression import ClassifierLogisticRegression
from learning.ClassifierLogisticRegression import OptionsLogisticRegression
from learning.ClassifierNN import ClassifierNN
from learning.ClassifierNN import OptionsNN
from learning.ClassifierSGD import OptionsSGD
from analyzing.ResultsAnalyzer import ResultsSingleConfigAnalyzer;
from analyzing.ResultsAnalyzer import ResultsAnalyzer

import helpers.constantsNZ as constantsNZ
import helpers.constants as constantsPATREC

def plotOneTrainingSetDifferentTestSets(results_analyzer, dirData, dirModelsBase, dirResultsBase):
    data_prefix = 'patrec'
    dict_options_dataset_training = {
        'dir_data':             dirData,
        'data_prefix':          data_prefix,
        'dataset':              '20122015',
        'options_filtering':    None
    }
    options_training = DatasetOptions(dict_options_dataset_training);
    # compare different subsets of data: EntlassBereich (only with RandomForest)
    options_rf = OptionsRF(dirModelsBase, options_training.getFilenameOptions(filteroptions=True));


    dict_options_dataset_testing = {
        'dir_data':             dirData,
        'data_prefix':          data_prefix,
        'dataset':              '20162017',
        'options_filtering':    None
    }
    options_testing_all = DatasetOptions(dict_options_dataset_testing);
    results_test_all = Results(dirResultsBase, options_training, options_rf, 'test', options_testing_all);


    dict_options_dataset_testing = {
        'dir_data':             dirData,
        'data_prefix':          data_prefix,
        'dataset':              '20162017',
        'options_filtering':    'EntlassBereich_Med'
    }
    options_testing_med = DatasetOptions(dict_options_dataset_testing);
    results_test_med = Results(dirResultsBase, options_training, options_rf, 'test', options_testing_med);

    dict_options_dataset_testing = {
        'dir_data':             dirData,
        'data_prefix':          data_prefix,
        'dataset':              '20162017',
        'options_filtering':    'EntlassBereich_SaO'
    }
    options_testing_sao = DatasetOptions(dict_options_dataset_testing);
    results_test_sao = Results(dirResultsBase, options_training, options_rf, 'test', options_testing_sao);

    dict_options_dataset_testing = {
        'dir_data':             dirData,
        'data_prefix':          data_prefix,
        'dataset':              '20162017',
        'options_filtering':    'EntlassBereich_Gyn'
    }
    options_testing_gyn = DatasetOptions(dict_options_dataset_testing);
    results_test_gyn = Results(dirResultsBase, options_training, options_rf, 'test', options_testing_gyn);

    analyzer_all = ResultsSingleConfigAnalyzer(results_test_all, 10);
    analyzer_med = ResultsSingleConfigAnalyzer(results_test_med, 10);
    analyzer_sao = ResultsSingleConfigAnalyzer(results_test_sao, 10);
    analyzer_gyn = ResultsSingleConfigAnalyzer(results_test_gyn, 10);
    analyzer = [analyzer_all, analyzer_med, analyzer_sao, analyzer_gyn];
    names = ['All', 'Med', 'SaO', 'Gyn']
    title_plot = 'classifier (rf): trained on patrec 2012-2015, tested on subsets of patrec 2016-2017'
    filename_plot = dirPlotsBase + 'rf_training_all_testing_EntlassBereich.png'
    results_analyzer.plotROCcurveMulitpleConfigs(analyzer, names, f_plot=filename_plot, titlePlot=title_plot, )


def plotDifferentTrainingSetDifferentTestSets(results_analyzer, dirData, dirModelsBase, dirResultsBase):
    data_prefix = 'patrec'

    # compare different subsets of data: EntlassBereich (only with RandomForest)
    dict_options_dataset_training = {
        'dir_data':             dirData,
        'data_prefix':          data_prefix,
        'dataset':              '20122015',
        'options_filtering':    None
    }
    dict_options_dataset_testing = {
        'dir_data':             dirData,
        'data_prefix':          data_prefix,
        'dataset':              '20162017',
        'options_filtering':    None
    }
    options_training_all = DatasetOptions(dict_options_dataset_training);
    options_testing_all = DatasetOptions(dict_options_dataset_testing);
    options_rf_all = OptionsRF(dirModelsBase, options_training_all.getFilenameOptions(filteroptions=True));
    results_test_all = Results(dirResultsBase, options_training_all, options_rf_all, 'test', options_testing_all);

    dict_options_dataset_training = {
        'dir_data':             dirData,
        'data_prefix':          data_prefix,
        'dataset':              '20122015',
        'options_filtering':    'EntlassBereich_Med'
    }
    dict_options_dataset_testing = {
        'dir_data':             dirData,
        'data_prefix':          data_prefix,
        'dataset':              '20162017',
        'options_filtering':    'EntlassBereich_Med'
    }
    options_training_med = DatasetOptions(dict_options_dataset_training);
    options_testing_med = DatasetOptions(dict_options_dataset_testing);
    options_rf_med = OptionsRF(dirModelsBase, options_training_med.getFilenameOptions(filteroptions=True));
    results_test_med = Results(dirResultsBase, options_training_med, options_rf_med, 'test', options_testing_med);

    dict_options_dataset_training = {
        'dir_data':             dirData,
        'data_prefix':          data_prefix,
        'dataset':              '20122015',
        'options_filtering':    'EntlassBereich_SaO'
    }
    dict_options_dataset_testing = {
        'dir_data':             dirData,
        'data_prefix':          data_prefix,
        'dataset':              '20162017',
        'options_filtering':    'EntlassBereich_SaO'
    }
    options_training_sao = DatasetOptions(dict_options_dataset_training);
    options_testing_sao = DatasetOptions(dict_options_dataset_testing);
    options_rf_sao = OptionsRF(dirModelsBase, options_training_sao.getFilenameOptions(filteroptions=True));
    results_test_sao = Results(dirResultsBase, options_training_sao, options_rf_sao, 'test', options_testing_sao);

    dict_options_dataset_training = {
        'dir_data':             dirData,
        'data_prefix':          data_prefix,
        'dataset':              '20122015',
        'options_filtering':    'EntlassBereich_Gyn'
    }
    dict_options_dataset_testing = {
        'dir_data':             dirData,
        'data_prefix':          data_prefix,
        'dataset':              '20162017',
        'options_filtering':    'EntlassBereich_Gyn'
    }
    options_training_gyn = DatasetOptions(dict_options_dataset_training);
    options_testing_gyn = DatasetOptions(dict_options_dataset_testing);
    options_rf_gyn = OptionsRF(dirModelsBase, options_training_gyn.getFilenameOptions(filteroptions=True));
    results_test_gyn = Results(dirResultsBase, options_training_gyn, options_rf_gyn, 'test', options_testing_gyn);

    analyzer_all = ResultsSingleConfigAnalyzer(results_test_all, 10);
    analyzer_med = ResultsSingleConfigAnalyzer(results_test_med, 10);
    analyzer_sao = ResultsSingleConfigAnalyzer(results_test_sao, 10);
    analyzer_gyn = ResultsSingleConfigAnalyzer(results_test_gyn, 10);
    analyzer = [analyzer_all, analyzer_med, analyzer_sao, analyzer_gyn];
    names = ['All', 'Med', 'SaO', 'Gyn']
    title_plot = 'classifier (rf): trained on subsets of patrec 2012-2015, tested on subsets of patrec 2016-2017'
    filename_plot = dirPlotsBase + 'rf_training_EntlassBereich_testing_EntlassBereich.png'
    results_analyzer.plotROCcurveMulitpleConfigs(analyzer, names, f_plot=filename_plot, titlePlot=title_plot, )


def plotDifferentTrainingSetSingleTestSetNZ(results_analyzer, dirData, dirModelsBase, dirResultsBase):
    print('plotDifferentTrainingSetSingleTestSetNZ')
    data_prefix = 'nz'

    dict_options_dataset_testing = {
        'dir_data':                 dirData,
        'data_prefix':              data_prefix,
        'dataset':                  '2017',
        'options_filtering':        None
    }
    options_testing = DatasetOptions(dict_options_dataset_testing);

    years_training = [2012, 2013, 2014, 2015, 2016];
    names = [];
    analyzers = []
    for year in years_training:
        print(year)
        dict_options_dataset_training = {
            'dir_data':             dirData,
            'data_prefix':          data_prefix,
            'dataset':              str(year),
            'options_filtering':    None
        }
        options_training_year = DatasetOptions(dict_options_dataset_training);
        options_rf_year = OptionsRF(dirModelsBase, options_training_year.getFilenameOptions(filteroptions=True));
        results_test_year = Results(dirResultsBase, options_training_year, options_rf_year, 'test', options_testing);

        names.append(str(year))
        analyzers.append(ResultsSingleConfigAnalyzer(results_test_year, 10));

    title_plot = 'classifier (rf): trained on subsets of nz 2012-2016, tested on subset of nz 2017'
    filename_plot = dirPlotsBase + 'rf_training_nz_years_20122016_testing_nz_year_2017.png'
    print('plot ROC curve...')
    results_analyzer.plotROCcurveMulitpleConfigs(analyzers, names, f_plot=filename_plot, titlePlot=title_plot, )


def plotDifferentClassifiers(results_analyzer, dirData, dirModelsBase, dirResultsBase):
    dict_opt_lr = {'penalty': 'l1', 'C': 0.5};
    dict_opt_rf = {'n_estimators': 500, 'max_depth': 50};
    dict_options_nn = {
        'hidden_units': [60, 40, 40, 20],
        'learningrate': 0.001,
        'dropout': 0.5,
        'batchnorm': True,
        'batch_size': 64,
        'training_epochs': 1000,
        'pretrained': None,
    }
    DIRPROJECT = '/home/thomas/projects/patrec';
    model_dir = os.path.join(DIRPROJECT, "patients_model")

    dict_options_dataset_training = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20122015',
        'grouping':         'verylightgrouping',
        'encoding':         'categorical',
        'newfeatures':      {'names': constantsPATREC.NEW_FEATURES},
        'featurereduction': None,
        'filtering':        None,
        'balanced':         False,
        'resample':         False
    }
    dict_options_dataset_testing = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20162017',
        'grouping':         'verylightgrouping',
        'encoding':         'categorical',
        'newfeatures':      {'names': constantsPATREC.NEW_FEATURES},
        'featurereduction': None,
        'filtering':        None,
        'balanced':         False,
        'resample':         False
    }

    options_training = DatasetOptions(dict_options_dataset_training);
    options_testing = DatasetOptions(dict_options_dataset_testing);

    dict_opt_rf = {'n_estimators': 500, 'max_depth': 50};
    options_rf = OptionsRF(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_opt_rf);
    results_test_rf = Results(dirResultsBase, options_training, options_rf, 'test', options_testing);

    dict_opt_lr_l2 = {'penalty': 'l2', 'C': 0.01};
    options_lr_l2 = OptionsLogisticRegression(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_opt_lr_l2);
    results_test_lr_l2 = Results(dirResultsBase, options_training, options_lr_l2, 'test', options_testing);

    dict_opt_lr_l1 = {'penalty': 'l1', 'C': 0.5};
    options_lr_l1 = OptionsLogisticRegression(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_opt_lr_l1);
    results_test_lr_l1 = Results(dirResultsBase, options_training, options_lr_l1, 'test', options_testing);

    dict_options_dataset_training_nn = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20122015',
        'grouping':         'verylightgrouping',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': None,
        'filtering':        None,
        'balanced':         False,
        'resample':         False
    }
    dict_options_dataset_testing_nn = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20162017',
        'grouping':         'verylightgrouping',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': None,
        'filtering':        None,
        'balanced':         False,
        'resample':         False
    }
    options_training = DatasetOptions(dict_options_dataset_training_nn);
    options_testing = DatasetOptions(dict_options_dataset_testing_nn);
    options_nn = OptionsNN(model_dir, options_training, options_clf=dict_options_nn);
    results_test_nn = Results(dirResultsBase, options_training, options_nn, 'test', options_testing);

    analyzer_rf = ResultsSingleConfigAnalyzer(results_test_rf, 10);
    analyzer_lr_l1 = ResultsSingleConfigAnalyzer(results_test_lr_l1, 10);
    analyzer_nn = ResultsSingleConfigAnalyzer(results_test_nn, 10)
    analyzer = [analyzer_rf, analyzer_lr_l1, analyzer_nn];
    names = ['RF', 'Logistic Regression (l1)', 'Neural Network']
    title_plot = ''
    filename_plot = os.path.join(dirPlotsBase,
                                 'different_classifiers_train_patrec_20122015_test_patrec_20162017.png')
    results_analyzer.plotROCcurveMulitpleConfigs(analyzer, names, f_plot=filename_plot, titlePlot=title_plot, )


def plotNNPerformance(results_analyzer, dirData, dirModelsBase, dirResultsBase):

    # compare different trainings of NNs
    dict_options_dataset_training = {
        'dir_data':             dirData,
        'data_prefix':          'nz',
        'dataset':              '20122016',
        'encoding':             'embedding',
        'newfeatures':          None,
        'featurereduction':     {'method': 'FUSION'},
        'grouping':             'verylightgrouping'
    }
    dict_options_dataset_testing = {
        'dir_data':             dirData,
        'data_prefix':          'nz',
        'dataset':              '2017',
        'encoding':             'embedding',
        'newfeatures':          None,
        'featurereduction':     {'method': 'FUSION'},
        'grouping':             'verylightgrouping'
    }
    options_training_nn = DatasetOptions(dict_options_dataset_training);
    options_testing_nn = DatasetOptions(dict_options_dataset_testing);

    dict_options_nn = {
        'hidden_units':     [60, 40, 20, 10, 10],
        'learningrate':     0.05,
        'dropout':          0.25,
        'batch_size':       640,
        'training_epochs':  250,
        'pretrained':       'pretrained'
    }
    options_nn_nz = OptionsNN(dirModelsBase, options_training_nn.getFilenameOptions(filteroptions=True), options_clf=dict_options_nn)
    results_nn_nz = Results(dirResultsBase, options_training_nn, options_nn_nz, 'test', options_testing_nn);

    dict_options_dataset_training = {
        'dir_data':             dirData,
        'data_prefix':          'patrec',
        'dataset':              '20122015',
        'encoding':             'embedding',
        'newfeatures':          None,
        'featurereduction':     {'method': 'FUSION'},
        'grouping':             'verylightgrouping'
    }
    dict_options_dataset_testing = {
        'dir_data':             dirData,
        'data_prefix':          'patrec',
        'dataset':              '20162017',
        'encoding':             'embedding',
        'newfeatures':          None,
        'featurereduction':     {'method': 'FUSION'},
        'grouping':             'verylightgrouping'
    }
    options_training_nn = DatasetOptions(dict_options_dataset_training);
    options_testing_nn = DatasetOptions(dict_options_dataset_testing);

    dict_options_nn = {
        'hidden_units':     [20, 10, 10],
        'learningrate':     0.01,
        'dropout':          0.15,
        'batch_size':       80,
        'training_epochs':  500,
    }
    options_nn_patrec = OptionsNN(dirModelsBase, options_training_nn.getFilenameOptions(filteroptions=True), options_clf=dict_options_nn)
    results_nn_patrec = Results(dirResultsBase, options_training_nn, options_nn_patrec, 'test', options_testing_nn);

    dict_options_nn = {
        'hidden_units':     [20, 10, 10],
        'learningrate':     0.01,
        'dropout':          0.25,
        'batch_size':       80,
        'training_epochs':  500,
        'pretrained':       'pretrained'
    }
    options_nn_patrec_pretrained = OptionsNN(dirModelsBase, options_training_nn.getFilenameOptions(filteroptions=True), options_clf=dict_options_nn)
    results_nn_patrec_pretrained = Results(dirResultsBase, options_training_nn, options_nn_patrec_pretrained, 'test', options_testing_nn);

    dict_options_dataset_training = {
        'dir_data': dirData,
        'data_prefix': 'patrec',
        'dataset': '20122015',
        'subgroups': ['DK'],
        'encoding': 'categorical',
        'newfeatures': None,
        'featurereduction': {'method': 'FUSION'},
        'grouping': 'verylightgrouping'
    }
    dict_options_dataset_testing = {
        'dir_data': dirData,
        'data_prefix': 'patrec',
        'dataset': '20162017',
        'subgroups': ['DK'],
        'encoding': 'categorical',
        'newfeatures': None,
        'featurereduction': {'method': 'FUSION'},
        'grouping': 'verylightgrouping'
    }
    dict_opt_lr = {'penalty': 'l1', 'C': 0.075};
    options_training = DatasetOptions(dict_options_dataset_training);
    options_testing = DatasetOptions(dict_options_dataset_testing);
    options_lr = OptionsLogisticRegression(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_opt_lr);
    results_lr = Results(dirResultsBase, options_training, options_lr, 'test', options_testing);

    analyzer_nn_nz = ResultsSingleConfigAnalyzer(results_nn_nz, 1);
    analyzer_nn_patrec = ResultsSingleConfigAnalyzer(results_nn_patrec, 1);
    analyzer_nn_patrec_pretrained = ResultsSingleConfigAnalyzer(results_nn_patrec_pretrained, 1);
    analyzer_lr = ResultsSingleConfigAnalyzer(results_lr, 10);
    analyzer = [analyzer_nn_nz, analyzer_nn_patrec, analyzer_nn_patrec_pretrained, analyzer_lr];
    names = ['NZ', 'Basel', 'Basel (pretrained NZ)', 'LASSO']
    title_plot = 'neural network performance: with and without pre-training'
    filename_plot = dirPlotsBase + 'nn_pretraining_nz_plus_lasso.png'
    results_analyzer.plotROCcurveMulitpleConfigs(analyzer, names, f_plot=filename_plot)


def plotOEPerformances(results_analyzer, dirData, dirModelsBase, dirResultsBase):
    dict_opt_lr = {'penalty': 'l1', 'C': 0.5};
    dict_opt_rf = {'n_estimators': 500, 'max_depth': 50};
    dict_options_nn = {
        'hidden_units':     [60, 40, 40, 20],
        'learningrate':     0.001,
        'dropout':          0.5,
        'batchnorm':        True,
        'batch_size':       64,
        'training_epochs':  1000,
        'pretrained':       None,
    }
    DIRPROJECT = '/home/thomas/projects/patrec';
    model_dir = os.path.join(DIRPROJECT, "patients_model")

    dict_options_dataset_training = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20122015',
        'grouping':         'verylightgrouping',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': None,
        'filtering':        None,
        'balanced':         False,
        'resample':         False
    }
    dict_options_dataset_testing = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20162017',
        'grouping':         'verylightgrouping',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': None,
        'filtering':        None,
        'balanced':         False,
        'resample':         False
    }
    dataset_options_training_all = DatasetOptions(dict_options_dataset_training);
    dataset_options_testing_all = DatasetOptions(dict_options_dataset_testing);
    options_all_nn = OptionsNN(model_dir, dataset_options_training_all, options_clf=dict_options_nn);
    options_all_lr = OptionsLogisticRegression(dirModelsBase,
                                                dataset_options_training_all.getFilenameOptions(filteroptions=True),
                                                options_clf=dict_opt_lr)
    options_all_rf = OptionsRF(dirModelsBase, dataset_options_training_all.getFilenameOptions(filteroptions=True),
                               options_clf=dict_opt_rf);
    classifier_nn_all = ClassifierNN(options_all_nn)
    classifier_lr_all = ClassifierLogisticRegression(options_all_lr)
    classifier_rf_all = ClassifierRF(options_all_rf)

    dict_options_dataset_training = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20122015',
        'grouping':         'verylightgrouping',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': None,
        'filtering':        'EntlassBereich_SaO',
        'balanced':         False,
        'resample':         False
    }
    dict_options_dataset_testing = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20162017',
        'grouping':         'verylightgrouping',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': None,
        'filtering':        'EntlassBereich_SaO',
        'balanced':         False,
        'resample':         False
    }
    dataset_options_training_SaO = DatasetOptions(dict_options_dataset_training);
    dataset_options_testing_SaO = DatasetOptions(dict_options_dataset_testing);
    options_SaO_nn = OptionsNN(model_dir, dataset_options_training_SaO, options_clf=dict_options_nn);
    options_SaO_lr = OptionsLogisticRegression(dirModelsBase,
                                               dataset_options_training_SaO.getFilenameOptions(filteroptions=True),
                                               options_clf=dict_opt_lr);
    options_SaO_rf = OptionsRF(dirModelsBase,
                               dataset_options_training_SaO.getFilenameOptions(filteroptions=True),
                               options_clf=dict_opt_rf)
    classifier_nn_SaO = ClassifierNN(options_SaO_nn);
    classifier_lr_SaO = ClassifierLogisticRegression(options_SaO_lr);
    classifier_rf_SaO = ClassifierRF(options_SaO_rf);

    dict_options_dataset_training = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20122015',
        'grouping':         'verylightgrouping',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': None,
        'filtering':        'EntlassBereich_Med',
        'balanced':         False,
        'resample':         False
    }
    dict_options_dataset_testing = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20162017',
        'grouping':         'verylightgrouping',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': None,
        'filtering':        'EntlassBereich_Med',
        'balanced':         False,
        'resample':         False
    }
    dataset_options_training_Med = DatasetOptions(dict_options_dataset_training);
    dataset_options_testing_Med = DatasetOptions(dict_options_dataset_testing);
    options_Med_nn = OptionsNN(model_dir, dataset_options_training_Med, options_clf=dict_options_nn);
    options_Med_lr = OptionsLogisticRegression(dirModelsBase,
                                               dataset_options_training_Med.getFilenameOptions(filteroptions=True),
                                               options_clf=dict_opt_lr);
    options_Med_rf = OptionsRF(dirModelsBase, dataset_options_training_Med.getFilenameOptions(filteroptions=True),
                               options_clf=dict_opt_rf)
    classifier_nn_Med = ClassifierNN(options_Med_nn)
    classifier_lr_Med = ClassifierLogisticRegression(options_Med_lr);
    classifier_rf_Med = ClassifierRF(options_Med_rf);

    dict_options_dataset_training = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20122015',
        'grouping':         'verylightgrouping',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': None,
        'filtering':        'EntlassBereich_Gyn',
        'balanced':         False,
        'resample':         False
    }
    dict_options_dataset_testing = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20162017',
        'grouping':         'verylightgrouping',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': None,
        'filtering':        'EntlassBereich_Gyn',
        'balanced':         False,
        'resample':         False
    }
    dataset_options_training_Gyn = DatasetOptions(dict_options_dataset_training);
    dataset_options_testing_Gyn = DatasetOptions(dict_options_dataset_testing);
    options_Gyn_nn = OptionsNN(model_dir, dataset_options_training_Gyn, options_clf=dict_options_nn);
    options_Gyn_lr = OptionsLogisticRegression(dirModelsBase,
                                               dataset_options_training_Gyn.getFilenameOptions(filteroptions=True),
                                               options_clf=dict_opt_lr);
    options_Gyn_rf = OptionsRF(dirModelsBase, dataset_options_training_Gyn.getFilenameOptions(filteroptions=True),
                               options_clf=dict_opt_rf);
    classifier_nn_Gyn = ClassifierNN(options_Gyn_nn)
    classifier_lr_Gyn = ClassifierLogisticRegression(options_Gyn_lr)
    classifier_rf_Gyn = ClassifierRF(options_Gyn_rf)

    results_all_nn = Results(dirResultsBase, dataset_options_training_all, options_all_nn, 'test', dataset_options_testing_all);
    results_SaO_nn = Results(dirResultsBase, dataset_options_training_SaO, options_SaO_nn, 'test', dataset_options_testing_SaO);
    results_Med_nn = Results(dirResultsBase, dataset_options_training_Med, options_Med_nn, 'test', dataset_options_testing_Med);
    results_Gyn_nn = Results(dirResultsBase, dataset_options_training_Gyn, options_Gyn_nn, 'test', dataset_options_testing_Gyn);
    results_all_lr = Results(dirResultsBase, dataset_options_training_all, options_all_lr, 'test', dataset_options_testing_all);
    results_SaO_lr = Results(dirResultsBase, dataset_options_training_SaO, options_SaO_lr, 'test', dataset_options_testing_SaO);
    results_Med_lr = Results(dirResultsBase, dataset_options_training_Med, options_Med_lr, 'test', dataset_options_testing_Med);
    results_Gyn_lr = Results(dirResultsBase, dataset_options_training_Gyn, options_Gyn_lr, 'test', dataset_options_testing_Gyn);
    results_all_rf = Results(dirResultsBase, dataset_options_training_all, options_all_rf, 'test', dataset_options_testing_all);
    results_SaO_rf = Results(dirResultsBase, dataset_options_training_SaO, options_SaO_rf, 'test', dataset_options_testing_SaO);
    results_Med_rf = Results(dirResultsBase, dataset_options_training_Med, options_Med_rf, 'test', dataset_options_testing_Med);
    results_Gyn_rf = Results(dirResultsBase, dataset_options_training_Gyn, options_Gyn_rf, 'test', dataset_options_testing_Gyn);

    analyzer_all_nn = ResultsSingleConfigAnalyzer(results_all_nn, 10);
    analyzer_SaO_nn = ResultsSingleConfigAnalyzer(results_SaO_nn, 10);
    analyzer_Med_nn = ResultsSingleConfigAnalyzer(results_Med_nn, 10);
    analyzer_Gyn_nn = ResultsSingleConfigAnalyzer(results_Gyn_nn, 10);
    analyzer_all_lr = ResultsSingleConfigAnalyzer(results_all_lr, 10);
    analyzer_SaO_lr = ResultsSingleConfigAnalyzer(results_SaO_lr, 10);
    analyzer_Med_lr = ResultsSingleConfigAnalyzer(results_Med_lr, 10);
    analyzer_Gyn_lr = ResultsSingleConfigAnalyzer(results_Gyn_lr, 10);
    analyzer_all_rf = ResultsSingleConfigAnalyzer(results_all_rf, 10);
    analyzer_SaO_rf = ResultsSingleConfigAnalyzer(results_SaO_rf, 10);
    analyzer_Med_rf = ResultsSingleConfigAnalyzer(results_Med_rf, 10);
    analyzer_Gyn_rf = ResultsSingleConfigAnalyzer(results_Gyn_rf, 10);

    analyzer_nn = [analyzer_all_nn, analyzer_Med_nn, analyzer_SaO_nn, analyzer_Gyn_nn];
    analyzer_lr = [analyzer_all_lr, analyzer_Med_lr, analyzer_SaO_lr, analyzer_Gyn_lr];
    analyzer_rf = [analyzer_all_rf, analyzer_Med_rf, analyzer_SaO_rf, analyzer_Gyn_rf];
    analyzer = analyzer_nn;
    names_nn = ['NN - all', 'NN - Med', 'NN - SaO', 'NN - Gyn']
    names_lr = ['LR - all', 'LR - Med', 'LR - SaO', 'LR - Gyn']
    names_rf = ['RF - all', 'RF - Med', 'RF - SaO', 'RF - Gyn']
    names = names_nn;
    title_plot = ''
    filename_plot_nn = os.path.join(dirPlotsBase, 'oes_nn_classification_performance.png')
    filename_plot_lr = os.path.join(dirPlotsBase, 'oes_lr_classification_performance.png')
    filename_plot_rf = os.path.join(dirPlotsBase, 'oes_rf_classification_performance.png')
    filename_plot = filename_plot_nn;
    results_analyzer.plotROCcurveMulitpleConfigs(analyzer, names, titlePlot=title_plot, f_plot=filename_plot)


def plotDiseasePerformances(results_analyzer, dirData, dirModelsBase, dirResultsBase):
    dict_opt_lr = {'penalty': 'l1', 'C': 0.5};
    dict_opt_rf = {'n_estimators': 500, 'max_depth': 50};
    dict_options_nn = {
        'hidden_units': [60, 40, 40, 20],
        'learningrate': 0.001,
        'dropout': 0.5,
        'batchnorm': True,
        'batch_size': 64,
        'training_epochs': 1000,
        'pretrained': None,
    }
    DIRPROJECT = '/home/thomas/projects/patrec';
    model_dir = os.path.join(DIRPROJECT, "patients_model")

    dict_options_dataset_training = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20122015',
        'grouping':         'verylightgrouping',
        'encoding':         'categorical',
        'newfeatures':      {'names': constantsPATREC.NEW_FEATURES},
        'featurereduction': None,
        'filtering':        None,
        'balanced':         False,
        'resample':         False
    }
    dict_options_dataset_testing = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20162017',
        'encoding':         'categorical',
        'newfeatures':      {'names': constantsPATREC.NEW_FEATURES},
        'grouping':         'verylightgrouping',
        'featurereduction': None,
        'filtering':        None,
        'balanced':         False,
        'resample':         False
    }

    dict_options_all_training = dict_options_dataset_training.copy();
    dict_options_all_testing = dict_options_dataset_testing.copy();
    options_all_training = DatasetOptions(dict_options_all_training);
    options_all_testing = DatasetOptions(dict_options_all_testing);
    options_all_lr = OptionsLogisticRegression(dirModelsBase,
                                               options_all_training.getFilenameOptions(filteroptions=True),
                                               options_clf=dict_opt_lr);
    options_all_rf = OptionsRF(dirModelsBase,
                               options_all_training.getFilenameOptions(filteroptions=True),
                               options_clf=dict_opt_rf);
    options_all_nn = OptionsNN(model_dir, options_all_training, options_clf=dict_options_nn);


    dict_options_lung_training = dict_options_dataset_training.copy();
    dict_options_lung_testing = dict_options_dataset_testing.copy();
    dict_options_lung_training['filtering'] = 'chronic_lung';
    dict_options_lung_testing['filtering'] = 'chronic_lung';
    options_lung_training = DatasetOptions(dict_options_lung_training);
    options_lung_testing = DatasetOptions(dict_options_lung_testing);
    options_lung_lr = OptionsLogisticRegression(dirModelsBase,
                                                options_lung_training.getFilenameOptions(filteroptions=True),
                                                options_clf=dict_opt_lr);
    options_lung_rf = OptionsRF(dirModelsBase,
                                options_lung_training.getFilenameOptions(filteroptions=True),
                                options_clf=dict_opt_rf);
    options_lung_nn = OptionsNN(model_dir, options_lung_training, options_clf=dict_options_nn);

    dict_options_oncology_training = dict_options_dataset_training.copy();
    dict_options_oncology_testing = dict_options_dataset_testing.copy();
    dict_options_oncology_training['filtering'] = 'oncology';
    dict_options_oncology_testing['filtering'] = 'oncology';
    options_oncology_training = DatasetOptions(dict_options_oncology_training);
    options_oncology_testing = DatasetOptions(dict_options_oncology_testing);
    options_oncology_lr = OptionsLogisticRegression(dirModelsBase,
                                                    options_oncology_training.getFilenameOptions(filteroptions=True),
                                                    options_clf=dict_opt_lr);
    options_oncology_rf = OptionsRF(dirModelsBase,
                                    options_oncology_training.getFilenameOptions(filteroptions=True),
                                    options_clf=dict_opt_rf);
    options_oncology_nn = OptionsNN(model_dir, options_oncology_training, options_clf=dict_options_nn);

    dict_options_cardio_training = dict_options_dataset_training.copy();
    dict_options_cardio_testing = dict_options_dataset_testing.copy();
    dict_options_cardio_training['filtering'] = 'cardiovascular';
    dict_options_cardio_testing['filtering'] = 'cardiovascular';
    options_cardio_training = DatasetOptions(dict_options_cardio_training);
    options_cardio_testing = DatasetOptions(dict_options_cardio_testing);
    options_cardio_lr = OptionsLogisticRegression(dirModelsBase,
                                                  options_cardio_training.getFilenameOptions(filteroptions=True),
                                                  options_clf=dict_opt_lr);
    options_cardio_rf = OptionsRF(dirModelsBase,
                                  options_cardio_training.getFilenameOptions(filteroptions=True),
                                  options_clf=dict_opt_rf);
    options_cardio_nn = OptionsNN(model_dir, options_cardio_training, options_clf=dict_options_nn);

    results_all_rf = Results(dirResultsBase, options_all_training, options_all_rf, 'test', options_all_testing);
    results_lung_rf = Results(dirResultsBase, options_lung_training, options_lung_rf, 'test', options_lung_testing);
    results_oncology_rf = Results(dirResultsBase, options_oncology_training, options_oncology_rf, 'test', options_oncology_testing);
    results_cardio_rf = Results(dirResultsBase, options_cardio_training, options_cardio_rf, 'test', options_cardio_testing);
    results_all_lr = Results(dirResultsBase, options_all_training, options_all_lr, 'test', options_all_testing);
    results_lung_lr = Results(dirResultsBase, options_lung_training, options_lung_lr, 'test', options_lung_testing);
    results_oncology_lr = Results(dirResultsBase, options_oncology_training, options_oncology_lr, 'test', options_oncology_testing);
    results_cardio_lr = Results(dirResultsBase, options_cardio_training, options_cardio_lr, 'test', options_cardio_testing);
    results_all_nn = Results(dirResultsBase, options_all_training, options_all_nn, 'test', options_all_testing);
    results_lung_nn = Results(dirResultsBase, options_lung_training, options_lung_nn, 'test', options_lung_testing);
    results_oncology_nn = Results(dirResultsBase, options_oncology_training, options_oncology_nn, 'test', options_oncology_testing);
    results_cardio_nn = Results(dirResultsBase, options_cardio_training, options_cardio_nn, 'test', options_cardio_testing);

    analyzer_all_rf = ResultsSingleConfigAnalyzer(results_all_rf, 10);
    analyzer_lung_rf = ResultsSingleConfigAnalyzer(results_lung_rf, 10);
    analyzer_oncology_rf = ResultsSingleConfigAnalyzer(results_oncology_rf, 10);
    analyzer_cardio_rf = ResultsSingleConfigAnalyzer(results_cardio_rf, 10);
    analyzer_all_lr = ResultsSingleConfigAnalyzer(results_all_lr, 10);
    analyzer_lung_lr = ResultsSingleConfigAnalyzer(results_lung_lr, 10);
    analyzer_oncology_lr = ResultsSingleConfigAnalyzer(results_oncology_lr, 10);
    analyzer_cardio_lr = ResultsSingleConfigAnalyzer(results_cardio_lr, 10);
    analyzer_all_nn = ResultsSingleConfigAnalyzer(results_all_nn, 10);
    analyzer_lung_nn = ResultsSingleConfigAnalyzer(results_lung_nn, 10);
    analyzer_oncology_nn = ResultsSingleConfigAnalyzer(results_oncology_nn,  10);
    analyzer_cardio_nn = ResultsSingleConfigAnalyzer(results_cardio_nn, 10);
    analyzer_rf = [analyzer_all_rf, analyzer_lung_rf, analyzer_oncology_rf, analyzer_cardio_rf];
    analyzer_lr = [analyzer_all_lr, analyzer_lung_lr, analyzer_oncology_lr, analyzer_cardio_lr];
    analyzer_nn = [analyzer_all_nn, analyzer_lung_nn, analyzer_oncology_nn, analyzer_cardio_nn]
    analyzer = analyzer_lr;

    names_rf = ['RF - all', 'RF - chronic lung', 'RF - oncology', 'RF - cardiovascular'];
    names_lr = ['LR - all', 'LR - chronic lung', 'LR - oncology', 'LR - cardiovascular'];
    names_nn = ['NN - all', 'NN - chronic lung', 'NN - oncology', 'NN - cardiovascular'];
    names = names_lr;

    title_plot = ''
    filename_plot_rf = os.path.join(dirPlotsBase, 'diseases_rf_classification_performance.png');
    filename_plot_lr = os.path.join(dirPlotsBase, 'diseases_lr_classification_performance.png');
    filename_plot_nn = os.path.join(dirPlotsBase, 'diseases_nn_classification_performance.png');
    filename_plot = filename_plot_lr;
    results_analyzer.plotROCcurveMulitpleConfigs(analyzer, names, titlePlot=title_plot, f_plot=filename_plot)


def plotSGDClassifierPerformance(results_analyzer, dirData, dirModelsBase, dirResultsBase):

    dict_options_dataset_testing = {
        'dir_data':         dirData,
        'data_prefix':      'nz',
        'dataset':          '2016',
        'encoding':         'categorical',
        'newfeatures':      {'names': constantsNZ.NEW_FEATURES},
        'featurereduction': None,
        'grouping':         'grouping'
    }
    options_dataset_testing = DatasetOptions(dict_options_dataset_testing);

    analyzer = [];
    years = [2012, 2013, 2014, 2015];
    for year in years:
        dict_options_dataset_training = {
            'dir_data':         dirData,
            'data_prefix':      'nz',
            'dataset':          str(year),
            'encoding':         'categorical',
            'newfeatures':      {'names': constantsNZ.NEW_FEATURES},
            'featurereduction': None,
            'grouping':         'grouping'
        }
        options_dataset_training = DatasetOptions(dict_options_dataset_training);

        dict_opt_sgd = {'loss': 'log', 'penalty': 'l1'};
        options_sgd = OptionsSGD(dirModelsBase, options_dataset_training.getFilenameOptions(filteroptions=True),options_clf=dict_opt_sgd);
        results_year = Results(dirResultsBase, options_dataset_training, options_sgd, 'test', options_dataset_testing);
        analyzer_sgd_year = ResultsSingleConfigAnalyzer(results_year, 1);
        analyzer.append(analyzer_sgd_year);

    names = ['2012', '2013', '2014', '2015'];
    title_plot = 'performance of batch-based logistic regression'
    filename_plot = dirPlotsBase + 'sgd_nz_performance_years_training20122015_test2016.png'
    results_analyzer.plotROCcurveMulitpleConfigs(analyzer, names, f_plot=filename_plot, titlePlot=title_plot)



def plotSingleConfiguration(results_analyzer, dirData, dirModelsBase, dirResultsBase):
    dict_options_dataset_training = {
        'dir_data': dirData,
        'data_prefix': 'patrec',
        'dataset': '20122015'
    }
    dict_options_dataset_testing = {
        'dir_data': dirData,
        'data_prefix': 'patrec',
        'dataset': '20162017'
    }

    options_training = DatasetOptions(dict_options_dataset_training);
    options_testing = DatasetOptions(dict_options_dataset_testing);
    options_rf = OptionsRF(dirModelsBase, options_training.getFilenameOptions(filteroptions=True));
    results_all_runs_test = Results(dirResultsBase, options_training, options_rf, 'test', options_testing);

    analyzer_single_config = ResultsSingleConfigAnalyzer(results_all_runs_test, 10);
    results_analyzer.plotROCcurveSingleConfig(analyzer_single_config, 'rf')


if __name__ == '__main__':
    # dirProject = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
    dirProject = '/home/thomas/fusessh/scicore/projects/patrec';
    dirData = os.path.join(dirProject,'data/');
    dirResultsBase = os.path.join(dirProject, 'results/');
    dirModelsBase = os.path.join(dirProject, 'classifiers/');
    dirPlotsBase = os.path.join(dirProject, 'plots', 'performance_measures');

    if not os.path.exists(dirPlotsBase):
        os.makedirs(dirPlotsBase);

    results_analyzer = ResultsAnalyzer();

    # plotSingleConfiguration(results_analyzer, dirData, dirModelsBase, dirResultsBase);
    # plotOneTrainingSetDifferentTestSets(results_analyzer, dirData, dirModelsBase, dirResultsBase);
    # plotDifferentTrainingSetDifferentTestSets(results_analyzer, dirData, dirModelsBase, dirResultsBase)
    # plotDifferentTrainingSetSingleTestSetNZ(results_analyzer, dirData, dirModelsBase, dirResultsBase)


    # plotNNPerformance(results_analyzer, dirData, dirModelsBase, dirResultsBase);
    # plotSGDClassifierPerformance(results_analyzer, dirData, dirModelsBase, dirResultsBase);

    plotDifferentClassifiers(results_analyzer, dirData, dirModelsBase, dirResultsBase)
    plotDiseasePerformances(results_analyzer, dirData, dirModelsBase, dirResultsBase);
    plotOEPerformances(results_analyzer, dirData, dirModelsBase, dirResultsBase);
