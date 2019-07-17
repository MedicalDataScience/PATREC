import os
import sys
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


    def getPrecision(self):
        return self.precision;

    def getRecall(self):
        return self.recall;

    def getThresholdsPrecisionRecall(self):
        return self.thresholds_precision_recall;

    def getFMeasure(self):
        return self.fmeasure;

    def getAvgPrecision(self):
        return self.average_precision;

    def getTPR(self):
        return self.tpr;

    def getFPR(self):
        return self.fpr;

    def getThresholdsTprFpr(self):
        return self.thresholds_tpr_fpr;

    def getAUC(self):
        return self.roc_auc;


class Results:
    def __init__(self, dir_results, dataset_options_training, classifier_options, results_type, dataset_options_testing=None):
        self.training_dataset_options = dataset_options_training;
        self.testing_dataset_options = dataset_options_testing;
        self.classifier_options = classifier_options;

        self.results_all_runs = [];

        self.results_type = results_type;
        self.dir_results = self._getDirResults(dir_results);
        self.filename_options = self._getStrFilenameResults();


    def _getStrFilenameResultsTrain(self):
        strFilenameDatasetTraining = self.training_dataset_options.getFilenameOptions(filteroptions=True);
        strFilenameClassifier = self.classifier_options.getFilenameOptions();
        strFilenameResults = self.results_type + '_' + strFilenameDatasetTraining + '_' + strFilenameClassifier;
        return strFilenameResults;


    def _getStrFilenameResultsEval(self):
        strFilenameDatasetTraining = self.training_dataset_options.getFilenameOptions(filteroptions=True);
        strFilenameClassifier = self.classifier_options.getFilenameOptions();
        strFilenameResults = self.results_type + '_' + strFilenameDatasetTraining + '_' + strFilenameClassifier;
        return strFilenameResults;


    def _getStrFilenameResultsTest(self):
        strFilenameDatasetTraining = self.training_dataset_options.getFilenameOptions(filteroptions=True);
        strFilenameDatasetTesting = self.testing_dataset_options.getFilenameOptions(filteroptions=True);
        strFilenameClassifier = self.classifier_options.getFilenameOptions();
        strFilenameResults = self.results_type + '_' + strFilenameDatasetTraining + '_' + strFilenameDatasetTesting + '_' + strFilenameClassifier;
        return strFilenameResults;


    def _getStrFilenameResults(self):
        if self.results_type == 'train':
            strFilenameResults = self._getStrFilenameResultsTrain();
        elif self.results_type == 'eval':
            strFilenameResults = self._getStrFilenameResultsEval();
        elif self.results_type == 'test':
            strFilenameResults = self._getStrFilenameResultsTest();
        else:
            print('no valid results type selected...exit')
            sys.exit();
        return strFilenameResults;


    def _getDirResults(self, dir_results):
        str = os.path.join(dir_results, self.classifier_options.getName(), + self.results_type);
        if not os.path.exists(str):
            os.makedirs(str);
        return str;


    def _writeNumericListOfListToFile(self, numList, filename):
        file = open(filename, 'w');
        for list in numList:
            if len(list) > 0:
                file.write(str(list[0]));
                for k in range(1, len(list)):
                    file.write(',' + str(list[k]));
                file.write('\n');
        file.close();


    def _getPrecisionAllRuns(self):
        precision_all = [];
        for res in self.results_all_runs:
            precision_all.append(res.getPrecision());
        return precision_all;


    def _getRecallAllRuns(self):
        recall_all = [];
        for res in self.results_all_runs:
            recall_all.append(res.getRecall());
        return recall_all;


    def _getFMeasureAllRuns(self):
        fmeasure_all = [];
        for res in self.results_all_runs:
            fmeasure_all.append(res.getFMeasure());
        return fmeasure_all;


    def _getTprAllRuns(self):
        tpr_all = [];
        for res in self.results_all_runs:
            tpr_all.append(res.getTPR());
        return tpr_all;


    def _getFprAllRuns(self):
        fpr_all = [];
        for res in self.results_all_runs:
            fpr_all.append(res.getFPR());
        return fpr_all;


    def _getAvgPrecisionAllRuns(self):
        avgprecision_all = [];
        for res in self.results_all_runs:
            avgprecision_all.append(res.getAvgPrecision());
        return avgprecision_all;


    def _getAUCAllRuns(self):
        auc_all = [];
        for res in self.results_all_runs:
            auc_all.append(res.getAUC())
        return auc_all;


    def getFilenameResults(self, results_measure):
        filename = self.dir_results + self.filename_options + '_' + results_measure + '.txt';
        return filename;


    def getDirResults(self):
        return self.dir_results;


    def addResultsSingleRun(self, res):
        self.results_all_runs.append(res);


    def writeResultsToFileDataset(self):
        filename_precision = os.path.join(self.dir_results, self.filename_options + '_precision.txt');
        filename_recall = os.path.join(self.dir_results, self.filename_options + '_recall.txt');
        filename_fmeasure = os.path.join(self.dir_results, self.filename_options + '_fmeasure.txt');
        filename_tpr = os.path.join(self.dir_results, self.filename_options + '_tpr.txt');
        filename_fpr = os.path.join(self.dir_results, self.filename_options + '_fpr.txt');
        filename_auc = os.path.join(self.dir_results, self.filename_options + '_auc.txt');
        filename_avgprecision = os.path.join(self.dir_results, self.filename_options + '_avgprecision.txt');
        
        precision = self._getPrecisionAllRuns();
        recall = self._getRecallAllRuns();
        fmeasure = self._getFMeasureAllRuns();
        tpr = self._getTprAllRuns();
        fpr = self._getFprAllRuns();
        auc = self._getAUCAllRuns();
        avg_precision = self._getAvgPrecisionAllRuns();

        self._writeNumericListOfListToFile(precision, filename_precision);
        self._writeNumericListOfListToFile(recall, filename_recall);
        self._writeNumericListOfListToFile(fmeasure, filename_fmeasure);
        self._writeNumericListOfListToFile(tpr, filename_tpr);
        self._writeNumericListOfListToFile(fpr, filename_fpr);
        self._writeNumericListOfListToFile([auc], filename_auc)
        self._writeNumericListOfListToFile([avg_precision], filename_avgprecision)




