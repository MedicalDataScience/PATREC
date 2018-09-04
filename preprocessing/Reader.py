
import numpy as np

class PatrecReader:

    def __init__(self):
        return;

    def readHeadersFromFile(filename):
        with open(filename) as f:
            headers = f.read().splitlines()
        return headers;

    def readNumericListFromFile(filename):
        with open(filename) as f:
            lines = f.read().splitlines()

        numList = [float(i) for i in lines]
        return numList;

    def readNumericListOfListFromFile(filename):
        with open(filename) as f:
            lines = f.read().splitlines()

        listOfList = [];
        for line in lines:
            numList = [float(i) for i in line.split(',')]
            listOfList.append(numList)

        return listOfList;

    def readDataAndLabelsFromDisk(dirData, strFilename):
        filename_data = dirData + 'data_' + strFilename + '.npy';
        filename_labels = dirData + 'labels_' + strFilename + '.npy';
        try:
            data = np.load(filename_data);
            labels = np.load(filename_labels);
        except FileNotFoundError:
            print('data file could not be found...');
            print(filename_data)
            data = np.empty(shape=(0, 0));
            labels = np.empty(shape=(0, 0));
        return [data, labels];

    def readResultsFromFileDataset(self, dataset, dirResults, strFilenameIn):

        filename_tpr = dirResults + dataset + '_tpr_' + strFilenameIn + '.txt';
        filename_fpr = dirResults + dataset + '_fpr_' + strFilenameIn + '.txt';
        filename_precision = dirResults + dataset + '_precision_' + strFilenameIn + '.txt';
        filename_recall = dirResults + dataset + '_recall_' + strFilenameIn + '.txt';
        filename_fmeasure = dirResults + dataset + '_fmeasure_' + strFilenameIn + '.txt';
        filename_auc = dirResults + dataset + '_auc_' + strFilenameIn + '.txt';
        filename_avgprecision = dirResults + dataset + '_avgprecision_' + strFilenameIn + '.txt';

        try:
            tpr = self.readNumericListOfListFromFile(filename_tpr);
            fpr = self.readNumericListOfListFromFile(filename_fpr);
            precision = self.readNumericListOfListFromFile(filename_precision);
            recall = self.readNumericListOfListFromFile(filename_recall);
            fmeasure = self.readNumericListOfListFromFile(filename_fmeasure);
            auc = self.readNumericListOfListFromFile(filename_auc);
            avgprecision = self.readNumericListOfListFromFile(filename_avgprecision);
        except FileNotFoundError:
            tpr = None;
            fpr = None;
            precision = None;
            recall = None;
            fmeasure = None;
            auc = None;
            avgprecision = None;

        results = {'precision': precision, 'recall': recall, 'fmeasure': fmeasure, 'tpr': tpr, 'fpr': fpr, 'auc': auc,
                   'avg_precision': avgprecision};
        return results;


    def readResultsFromFile(self, dirResults, strFilename):
        results_training = self.readResultsFromFileDataset('training', dirResults, strFilename);
        results_testing = self.readResultsFromFileDataset('testing', dirResults, strFilename);
        return [results_training, results_testing];


    #TODO: probably it would be better to remove this function and just use readNumericListOfListFromFile
    def readFeatureCoefsFromFile(self, dirResults, strFilenameCoef):
        filename_feature = dirResults + 'featureweights_' + strFilenameCoef + '.txt';
        feature_coefs = self.readNumericListOfListFromFile(filename_feature);
        return feature_coefs;



