import os
import numpy as np
import matplotlib.pyplot as plt

class LearnedFeaturesAnalyzer:


    def __init__(self, dir_plots_base, options_training_data, options_classifier, dataset_training, num_runs):
        self.dataset_training = dataset_training;
        self.options_training_data = options_training_data;
        self.options_classifier = options_classifier;
        self.dir_plots = self._getDirPlots(dir_plots_base);
        self.num_runs = num_runs;


    def _getDirPlots(self, dir_plots_base):
        dir_plots = dir_plots_base + self.options_classifier.getName() +'/';
        if not os.path.exists(dir_plots):
            os.makedirs(dir_plots);
        return dir_plots;


    def _readNumericListFromFile(self, filename):
        with open(filename) as f:
            lines = f.read().splitlines()
        numList = [float(i) for i in lines]
        return numList;

    def _readFeatureCoefsFromFile(self, run):
        filename = self.options_classifier.getFilenameLearnedFeatures(run);
        feature_coefs = self._readNumericListFromFile(filename);
        return feature_coefs;


    def _readFeatureWeights(self):
        feature_names = self.dataset_training.getFinalColumns();
        feature_names.pop(feature_names.index('Wiederkehrer'))
        feature_weights = np.zeros((self.num_runs, len(feature_names)))
        for run in range(0, self.num_runs):
            feature_coefs = self._readFeatureCoefsFromFile(run);
            feature_weights[run,:] = feature_coefs;

        return [feature_weights, feature_names];


    def _getAvgFeatureWeights(self):
        [feature_weights_all_runs, feature_names] = self._readFeatureWeights();
        avg_weights = np.mean(feature_weights_all_runs, axis=0);
        return [avg_weights, feature_names];


    def getAvgFeatureWeights(self):
        [avg_weights, names] = self._getAvgFeatureWeights()
        return [avg_weights, names];

    def _getSortedAvgFeatureWeights(self, num_features):
        [avg_weights, names] = self._getAvgFeatureWeights()
        name_value_pairs = [];
        for k in range(0, len(names)):
            name_value_pairs.append([abs(avg_weights[k]), names[k]]);
        sorted_name_value_pairs = sorted(name_value_pairs, key=lambda x: x[0], reverse=True)[:num_features];

        weights = [];
        names = [];
        for k in range(0, num_features):
            weights.append(sorted_name_value_pairs[k][0]);
            names.append(sorted_name_value_pairs[k][1]);
        return [weights, names];

    def _createBarPlotForAxesObj(self, ax_obj, index_vec, values_vec, labels_vec, titleStr):
        ax_obj.bar(index_vec, values_vec, 0.75, align='center', color='b', alpha=0.5);
        ax_obj.xaxis.set_ticks(index_vec);
        ax_obj.set_xticklabels(labels_vec, rotation=90)
        ax_obj.xaxis.set_tick_params(labelsize=8)
        ax_obj.set_xlim([-1, len(index_vec)])
        ax_obj.set_title(titleStr)
        return ax_obj;


    def plotAvgLearnedFeatures(self, num_features=50):
        filename = self.dir_plots + 'learnedfeatures_avg_' + self.options_classifier.getFilenameOptions() + '_' + self.options_classifier.filename_options_training_data + '.png';
        [avg_weights, names] = self._getSortedAvgFeatureWeights(num_features);
        index_vec = range(0, num_features);
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7));
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.3, wspace=0.01, hspace=0.01)
        ax = self._createBarPlotForAxesObj(ax, index_vec, avg_weights, names, '');
        plt.draw()
        plt.savefig(filename, format='png');
        plt.show();






