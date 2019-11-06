import os
import numpy as np
import matplotlib.pyplot as plt

from utils.Dataset import Dataset

class LearnedFeaturesAnalyzer:


    def __init__(self, dir_plots_base, options_training_data, options_classifier, num_runs):
        self.options_training_data = options_training_data;
        self.dataset_training = Dataset(self.options_training_data);
        self.options_classifier = options_classifier;
        self.dir_plots = self._getDirPlots(dir_plots_base);
        self.num_runs = num_runs;


    def _getDirPlots(self, dir_plots_base):
        dir_plots = os.path.join(dir_plots_base, self.options_classifier.getName() +'/');
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
        early_readmission_flagname = self.options_training_data.getEarlyReadmissionFlagname();
        feature_names = self.dataset_training.getColumnsData();
        feature_names.pop(feature_names.index(early_readmission_flagname))
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

    def _switchToEnglishNames(self, names):
        english_names = []
        for name in names:
            if name.startswith('Liegestatus'):
                new_name = 'LOS_';
                cat_value = name.split('_')[-1]
                if cat_value == 'norm':
                    new_cat_value = 'inlier'
                elif cat_value == 'opti':
                    new_cat_value = 'optimal'
                elif cat_value == 'kurz':
                    new_cat_value = 'low'
                elif cat_value == 'high':
                    new_cat_value = 'high'
                elif cat_value == 'unb':
                    new_cat_value = 'unknown'
                else:
                    new_cat_value = cat_value
                new_name = new_name + new_cat_value
            elif name.startswith('Eintrittsalter'):
                new_name = 'Age'
            elif name.startswith('Aufnahmeart'):
                new_name = 'Admission_';
                cat_value = name.split('_')[-1]
                if cat_value == 'G Entbindung':
                    new_cat_value = 'childbirth';
                else:
                    new_cat_value = 'unknown';
                new_name = new_name + new_cat_value
            else:
                new_name = name
            english_names.append(new_name)
        return english_names

    def _createBarPlotForAxesObj(self, ax_obj, index_vec, values_vec, labels_vec, titleStr):
        values_vec = values_vec[::-1]
        labels_vec = labels_vec[::-1]
        ax_obj.barh(index_vec, values_vec, 0.75, align='center', color='b', alpha=0.5)
        ax_obj.yaxis.set_ticks(index_vec);
        ax_obj.set_yticklabels(labels_vec)
        ax_obj.yaxis.set_tick_params(labelsize=12)
        ax_obj.set_ylim([-1, len(index_vec)])
        ax_obj.set_title(titleStr)
        ax_obj.set_xlabel('Feature Weight')
        return ax_obj;

    def plotAvgLearnedFeatures(self, num_features=50, english=False):
        filename = os.path.join(self.dir_plots, 'learnedfeatures_avg_' + self.options_classifier.getFilenameOptions() + '_' + self.options_classifier.filename_options_training_data + '.png');
        print(filename)
        [avg_weights, names] = self._getSortedAvgFeatureWeights(num_features);
        print('start plotting...')
        index_vec = range(0, num_features);
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6));
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.1, left=0.4, hspace=0.01)
        if english:
            names = self._switchToEnglishNames(names)
        ax = self._createBarPlotForAxesObj(ax, index_vec, avg_weights, names, '');
        plt.draw()
        plt.savefig(filename, format='png');
        print('DONE')






