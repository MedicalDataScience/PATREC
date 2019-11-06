
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
import sklearn.metrics as metrics

class ResultsSingleConfigAnalyzer:

    def __init__(self, results, num_runs):
        self.results = results;
        self.num_runs = num_runs;

        self.precision_all_runs = None;
        self.recall_all_runs = None;
        self.thresholds_precision_recall_all_runs = None;
        self.avgprecision_all_runs = None;
        self.fmeasure_all_runs = None;
        self.tpr_all_runs = None;
        self.fpr_all_runs = None;
        self.thresholds_tpr_fpr_all_runs = None;
        self.roc_auc_all_runs = None;

        self.mean_precision = None;
        self.mean_recall = None;
        self.std_recall = None;
        self.mean_auc_precision_recall = None;
        self.std_auc_precision_recall = None;
        self.mean_avgprecision = None;
        self.std_avgprecision = None;
        self.mean_tpr = None;
        self.std_tpr = None;
        self.mean_fpr = None;
        self.mean_auc_tpr_fpr = None;
        self.std_auc_tpr_fpr = None;
        return;


    def _readNumericListOfListFromFile(self, filename):
        with open(filename) as f:
            lines = f.read().splitlines()
        listOfList = [];
        for line in lines:
            numList = [float(i) for i in line.split(',')]
            listOfList.append(numList)
        return listOfList;


    def _getValuesAllRuns(self, perf_measure):
        if perf_measure == 'precision':
            measures_all_runs = self.precision_all_runs;
        elif perf_measure == 'recall':
            measures_all_runs = self.recall_all_runs;
        elif perf_measure == 'fmeasure':
            measures_all_runs = self.fmeasure_all_runs;
        elif perf_measure == 'avgprecision':
            measures_all_runs = self.avgprecision_all_runs;
        elif perf_measure == 'tpr':
            measures_all_runs = self.tpr_all_runs;
        elif perf_measure == 'fpr':
            measures_all_runs = self.fpr_all_runs;
        elif perf_measure == 'auc':
            measures_all_runs = self.auc_all_runs;
        else:
            print('performance measure is unknown...exit!')
            sys.exit();
        return measures_all_runs;


    def _getMeanAndStdTprFpr(self):
        aucs = [];
        tprs = [];
        mean_fpr = np.linspace(0, 1, 100);
        # num runs
        for l in range(0, len(self.tpr_all_runs)):
            tprs.append(scipy.interp(mean_fpr, self.fpr_all_runs[l], self.tpr_all_runs[l]))
            tprs[-1][0] = 0.0
            roc_auc = metrics.auc(self.fpr_all_runs[l], self.tpr_all_runs[l])
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0);
        mean_tpr[-1] = 1.0;
        mean_auc = metrics.auc(mean_fpr, mean_tpr);
        std_auc = np.std(aucs);
        self.mean_tpr = mean_tpr;
        self.std_tpr = std_tpr;
        self.mean_fpr = mean_fpr;
        self.mean_auc_tpr_fpr = mean_auc;
        self.std_auc_tpr_fpr = std_auc;

    def _getMeanAndStdPrecisionRecall(self):
        recs = [];
        aucs = [];
        mean_p = np.linspace(0, 1, 100);
        # num runs
        for l in range(0, len(self.precision_all_runs)):
            recs.append(scipy.interp(mean_p, self.precision_all_runs[l], self.recall_all_runs[l]))
            recs[-1][0] = 1.0
            roc_auc = metrics.auc(self.recall_all_runs[l], self.precision_all_runs[l])
            aucs.append(roc_auc)

        mean_r = np.mean(recs, axis=0)
        mean_r[-1] = 0.0;
        mean_auc = metrics.auc(mean_p, mean_r);
        std_auc = np.std(aucs);
        mean_avgprecision = np.mean(self.avgprecision_all_runs);
        std_avgprecision = np.std(self.avgprecision_all_runs);
        self.mean_precision = mean_p;
        self.mean_recall = mean_r;
        self.std_recall = np.std(recs);
        self.mean_auc_precision_recall = mean_auc;
        self.std_auc_precision_recall = std_auc;
        self.mean_avgprecision = mean_avgprecision;
        self.std_avgprecision = std_avgprecision;


    def readResultsFromFile(self):
        filename_precision = self.results.getFilenameResults('precision');
        filename_recall = self.results.getFilenameResults('recall');
        filename_fmeasure = self.results.getFilenameResults('fmeasure');
        filename_tpr = self.results.getFilenameResults('tpr');
        filename_fpr = self.results.getFilenameResults('fpr');
        filename_auc = self.results.getFilenameResults('auc');
        filename_avgprecision = self.results.getFilenameResults('avgprecision');
        self.precision_all_runs = self._readNumericListOfListFromFile(filename_precision);
        self.recall_all_runs = self._readNumericListOfListFromFile(filename_recall);
        self.fmeasure_all_runs = self._readNumericListOfListFromFile(filename_fmeasure);
        self.tpr_all_runs = self._readNumericListOfListFromFile(filename_tpr);
        self.fpr_all_runs = self._readNumericListOfListFromFile(filename_fpr);
        self.auc_all_runs = self._readNumericListOfListFromFile(filename_auc);
        self.avgprecision_all_runs = self._readNumericListOfListFromFile(filename_avgprecision);


    def calculateMeanAndStd(self):
        self._getMeanAndStdTprFpr();
        self._getMeanAndStdTprFpr();


    def getValuesAllRuns(self, perf_measure):
        return self._getValuesAllRuns(perf_measure);


    def getMeanFpr(self):
        if self.mean_fpr is None:
            self._getMeanAndStdTprFpr();
        return self.mean_fpr;

    def getMeanTpr(self):
        if self.mean_tpr is None:
            self._getMeanAndStdTprFpr();
        return self.mean_tpr;

    def getStdTpr(self):
        if self.std_tpr is None:
            self._getMeanAndStdTprFpr();
        return self.std_tpr;

    def getMeanAucFprTpr(self):
        if self.mean_auc_tpr_fpr is None:
            self._getMeanAndStdTprFpr();
        return self.mean_auc_tpr_fpr;

    def getStdAucFprTpr(self):
        if self.std_auc_tpr_fpr is None:
            self._getMeanAndStdTprFpr();
        return self.std_auc_tpr_fpr;


class ResultsAnalyzer:

    def __init__(self):
        return;


    def _plotTprFprCurve(self, res_analyzer, names, filename_plot=None, title_plot=None, show_std=False):

        num_different_configs = len(res_analyzer);
        colors = plt.cm.rainbow(np.linspace(0, 1, num_different_configs));

        plt.figure(figsize=(12, 10))
        for k in range(0, num_different_configs):
            col = colors[k, :];
            res = res_analyzer[k];
            mean_fpr_values = res.getMeanFpr();
            mean_tpr_values = res.getMeanTpr();
            std_tpr_values = res.getStdTpr();
            tprs_upper = np.minimum(mean_tpr_values + std_tpr_values, 1)
            tprs_lower = np.maximum(mean_tpr_values - std_tpr_values, 0)
            mean_auc = res.getMeanAucFprTpr();
            std_auc = res.getStdAucFprTpr();
            plt.step(mean_fpr_values, mean_tpr_values, c=col, linewidth=3,
                     label=r'%s (AUC = %0.2f $\pm$ %0.2f)' % (names[k], mean_auc, std_auc));
            if show_std:
                plt.fill_between(mean_fpr_values, tprs_lower, tprs_upper, color=col, alpha=.2);

        plt.xlabel('fpr');
        plt.ylabel('tpr');
        plt.xlim([0, 1]);
        plt.ylim([0, 1]);
        plt.legend(loc='lower right', prop={'size': 14});
        plt.grid(True)
        if title_plot is not None:
            plt.suptitle(title_plot)
        plt.draw();
        print(filename_plot)
        if filename_plot is not None:
            plt.savefig(filename_plot, format='png', bbox_inches='tight');
        plt.close();


    def plotROCcurveSingleConfig(self, res, name_config, f_plot=None, titlePlot=None):
        res.readResultsFromFile();
        res.calculateMeanAndStd();
        names = [name_config];
        self._plotTprFprCurve([res], names, filename_plot=f_plot, title_plot=titlePlot);


    def plotROCcurveMulitpleConfigs(self, list_res, names_config, f_plot=None, titlePlot=None, show_std=False):
        for res in list_res:
            res.readResultsFromFile();
            res.calculateMeanAndStd();
        self._plotTprFprCurve(res_analyzer=list_res, names=names_config, filename_plot=f_plot, title_plot=titlePlot, show_std=show_std)
