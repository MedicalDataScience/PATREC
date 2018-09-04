
import os

class PatrecWriter:

    def __init__(self):
        return;

    def writeListOfStringsToFile(strList, filename):

        file_out = open(filename, 'w');
        for str in strList:
            file_out.write("%s\n" % str)
        file_out.close();

    def writeNumericListToFile(numList, filename):

        file = open(filename, 'w');
        for num in numList:
            file.write(str(num) + '\n');
        file.close();

    def writeNumericListOfListToFile(numList, filename):
        file = open(filename, 'w');
        for list in numList:
            if len(list) > 0:
                file.write(str(list[0]));
                for k in range(1, len(list)):
                    file.write(',' + str(list[k]));
                file.write('\n');
        file.close();

    def writeResultsToFileDataset(self, dataset, results, dirResults, strFilenameOut):
        filename_precision = dirResults + dataset + '_precision_' + strFilenameOut + '.txt';
        filename_recall = dirResults + dataset + '_recall_' + strFilenameOut + '.txt'
        filename_fmeasure = dirResults + dataset + '_fmeasure_' + strFilenameOut + '.txt'
        filename_tpr = dirResults + dataset + '_tpr_' + strFilenameOut + '.txt'
        filename_fpr = dirResults + dataset + '_fpr_' + strFilenameOut + '.txt';
        filename_auc = dirResults + dataset + '_auc_' + strFilenameOut + '.txt';
        filename_avgprecision = dirResults + dataset + '_avgprecision_' + strFilenameOut + '.txt';

        precision = [];
        recall = [];
        fmeasure = [];
        tpr = [];
        fpr = [];
        auc = [];
        avg_precision = [];

        for k in range(0, len(results)):
            res = results[k];
            precision.append(res['precision']);
            recall.append(res['recall']);
            fmeasure.append(res['fmeasure']);
            tpr.append(res['tpr']);
            fpr.append(res['fpr']);
            auc.append(res['auc']);
            avg_precision.append(res['avg_precision']);

        self.writeNumericListOfListToFile(precision, filename_precision);
        self.writeNumericListOfListToFile(recall, filename_recall);
        self.writeNumericListOfListToFile(fmeasure, filename_fmeasure);
        self.writeNumericListOfListToFile(tpr, filename_tpr);
        self.writeNumericListOfListToFile(fpr, filename_fpr);
        self.writeNumericListOfListToFile([auc], filename_auc)
        self.writeNumericListOfListToFile([avg_precision], filename_avgprecision)


    def writeEvaluationResultsToFile(self, dirResults, strFilename, training_results, testing_results):

        dirResultsTraining = dirResults + 'train/';
        dirResultsEval = dirResults + 'eval/';

        if not os.path.exists(dirResultsTraining):
            os.makedirs(dirResultsTraining);

        if not os.path.exists(dirResultsEval):
            os.makedirs(dirResultsEval);

        self.writeResultsToFileDataset('train', training_results, dirResultsTraining, strFilename);
        self.writeResultsToFileDataset('eval', testing_results, dirResultsEval, strFilename);