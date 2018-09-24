
import numpy as np

class ResultsSingleRun:
    def __init__(self):

        self.precision = None;
        self.recall = None;
        self.thresholds_precision_recall = None;
        self.average_precision = None;
        self.fmeasure = None;

        self.tpr = None;
        self.fpr = None;
        self.thresholds_tpr_fpr = None;
        self.roc_auc = None;
        return;


    def _calcSingleFMeasureValue(self, precision, recall):
        if precision >= 0 and recall >= 0:
            try:
                fmeasure = float(2 * precision * recall) / float(precision + recall);
            except ZeroDivisionError:
                fmeasure = 0.0;
        else:
            fmeasure = -1;
        return fmeasure;


    def calcFMeasure(self, precisions, recalls):
        fmeasure = []
        for k in range(0, precisions.shape[0]):
            fmeasure.append(self._calcSingleFMeasureValue(precisions[k], recalls[k]));
        self.fmeasure = np.array(fmeasure);


    def setMetrics(self, results_dict):
        self.precision = results_dict['precision'];
        self.recall = results_dict['recall'];
        self.thresholds_precision_recall = results_dict['thresholds_precision_recall'];
        self.average_precision = results_dict['average_precision'];
        self.fmeasure = results_dict['fmeasure'];

        self.tpr = results_dict['tpr'];
        self.fpr = results_dict['fpr'];
        self.thresholds_tpr_fpr = results_dict['thresholds_tpr_fpr'];
        self.roc_auc = results_dict['roc_auc'];
        return;


    def readResults(self):
        return;

    def writeResults(self):
        return;


    def getAUC(self):
        return self.roc_auc;


class Results:
    def __init__(self, dataset_options, classifier_options):
        self.dataset_options = dataset_options;
        self.classifier_options = classifier_options;
        self.results_all_runs = [];


    def setResultsSingleRun(self, res):
        results_single_run = ResultsSingleRun();
        results_single_run.setMetrics(res);
        self.results_all_runs.append(results_single_run);




