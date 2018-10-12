import os
import numpy as np
from utils.DatasetOptions import DatasetOptions
from utils.Dataset import Dataset
from utils.Results import Results

from learning.ClassifierRF import OptionsRF
from learning.ClassifierLogisticRegression import ClassifierLogisticRegression
from learning.ClassifierLogisticRegression import OptionsLogisticRegression
from analyzing.ResultsAnalyzer import ResultsSingleConfigAnalyzer;
from analyzing.ResultsAnalyzer import ResultsAnalyzer



def plotOneTrainingSetDifferentTestSets(results_analyzer):
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


def plotDifferentTrainingSetDifferentTestSets(results_analyzer):
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


def plotDifferentClassifiers(results_analyzer):
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
    options_training = DatasetOptions(dict_options_dataset_training);
    options_testing = DatasetOptions(dict_options_dataset_testing);

    dict_opt_rf = {'n_estimators': 100, 'max_depth': 15};
    options_rf = OptionsRF(dirModelsBase, options_training.getFilenameOptions(filteroptions=True));
    results_test_rf = Results(dirResultsBase, options_training, options_rf, 'test', options_testing);

    dict_opt_lr_l2 = {'penalty': 'l2', 'C': 0.01};
    options_lr_l2 = OptionsLogisticRegression(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_opt_lr_l2);
    results_test_lr_l2 = Results(dirResultsBase, options_training, options_lr_l2, 'test', options_testing);

    dict_opt_lr_l1 = {'penalty': 'l1', 'C': 0.01};
    options_lr_l1 = OptionsLogisticRegression(dirModelsBase, options_training.getFilenameOptions(filteroptions=True), options_clf=dict_opt_lr_l1);
    results_test_lr_l1 = Results(dirResultsBase, options_training, options_lr_l1, 'test', options_testing);

    analyzer_rf = ResultsSingleConfigAnalyzer(results_test_rf, 10);
    analyzer_lr_l1 = ResultsSingleConfigAnalyzer(results_test_lr_l1, 10);
    analyzer_lr_l2 = ResultsSingleConfigAnalyzer(results_test_lr_l2, 10);
    analyzer = [analyzer_rf, analyzer_lr_l1, analyzer_lr_l2];
    names = ['rf', 'logistic regression (l1)', 'logistic regression (l2)']
    title_plot = 'multiple classifiers: trained on patrec 2012-2015, tested on patrec 2016-2017'
    filename_plot = dirPlotsBase + 'different_classifiers_train_patrec_20122015_test_patrec_20162017.png'
    results_analyzer.plotROCcurveMulitpleConfigs(analyzer, names, f_plot=filename_plot, titlePlot=title_plot, )





def plotSingleConfiguration(results_analyzer):
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
    dirProject = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
    dirData = dirProject + 'data/';
    dirResultsBase = dirProject + 'results/';
    dirModelsBase = dirProject + 'classifiers/'
    dirPlotsBase = dirProject + 'plots/performance_measures/';

    if not os.path.exists(dirPlotsBase):
        os.makedirs(dirPlotsBase);

    results_analyzer = ResultsAnalyzer();

    plotSingleConfiguration(results_analyzer);

    plotOneTrainingSetDifferentTestSets(results_analyzer);

    plotDifferentTrainingSetDifferentTestSets(results_analyzer)

    plotDifferentClassifiers(results_analyzer)



