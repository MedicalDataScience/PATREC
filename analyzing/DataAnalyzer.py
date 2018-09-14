
import matplotlib.pyplot as plt
import numpy as np

from analyzing.Dataset import Dataset

from helpers.helpers import getFeatureCategories

class DataAnalyzer:

    def __init__(self, dataset, dir_plots):
        self.dataset = dataset;
        self.dir_plots = dir_plots;
        return;


    def _printValues(self, category_names, occ_wiederkehrer, occ_normal):
        for k,name in enumerate(category_names):
            print(name + ': ' + str(occ_wiederkehrer[k]) + ' <-> ' + str(occ_normal[k]));


    def _getFeatureValues(self, df, name_feature):
        column_names = self.dataset.getColumns();
        feature_columns = [];
        for col in column_names:
            if col.startswith(name_feature):
                feature_columns.append(col);
        df_feature = df[feature_columns];
        df_feature_wiederkehrer = df_feature.loc[df['Wiederkehrer'] == 1];
        df_feature_normal = df_feature.loc[df['Wiederkehrer'] == 0];
        return [df_feature_normal, df_feature_wiederkehrer];


    # for categorical features
    def _doComparisonBar(self, df, name_feature):
        filename_plot = self.dir_plots + 'featurecomparison_' + name_feature + '.png';
        print(name_feature)
        categories_feature = getFeatureCategories(name_feature);
        print(categories_feature)
        values_to_count = range(0, len(categories_feature));

        [df_feature_normal, df_feature_wiederkehrer] = self._getFeatureValues(df, name_feature);
        num_feature_normal = df_feature_normal.shape[0];
        num_feature_wiederkehrer = df_feature_wiederkehrer.shape[0];
        occ_feature_wiederkehrer = df_feature_wiederkehrer.sum(axis=0);
        occ_feature_normal = df_feature_normal.sum(axis=0);

        self._printValues(categories_feature, occ_feature_wiederkehrer, occ_feature_normal);

        occ_wiederkehrer = occ_feature_wiederkehrer.values;
        occ_normal = occ_feature_normal.values;
        density_normal = occ_normal / float(num_feature_normal);
        density_wiederkehrer = occ_wiederkehrer / float(num_feature_wiederkehrer);

        print(len(values_to_count))
        print(density_wiederkehrer.shape)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 10));
        plt.bar(values_to_count, height=density_wiederkehrer.flatten(), width=1.0, align='center', color='b', alpha=0.5)
        plt.bar(values_to_count, height=density_normal.flatten(), width=1.0, align='center', color='m', alpha=0.5)
        plt.xlim([-1, len(categories_feature) + 1])
        plt.xticks(range(0, len(values_to_count)), categories_feature)
        plt.legend(['Wiederkehrer', 'normal'])
        plt.title(name_feature);
        plt.draw()
        plt.savefig(filename_plot, format='png')
        plt.close();


    # for numerical features
    def _doComparisonHist(self, df, name_feature):
        filename_plot = self.dir_plots + 'featurecomparison_' + name_feature + '.png';
        print(name_feature)

        [df_feature_normal, df_feature_wiederkehrer] = self._getFeatureValues(df, name_feature);
        num_values_normal = df_feature_normal.shape[0];
        num_values_wiederkehrer = df_feature_wiederkehrer.shape[0];
        values_wiederkehrer = df_feature_wiederkehrer.values;
        values_normal = df_feature_normal.values;

        if num_values_normal > 0 and num_values_wiederkehrer > 0:
            min_value = float(min(min(values_normal), min(values_wiederkehrer)));
            max_value = float(max(max(values_normal), max(values_wiederkehrer)));
        elif num_values_wiederkehrer > 0:
            min_value = float(min(values_wiederkehrer));
            max_value = float(max(values_wiederkehrer));
        elif num_values_normal > 0:
            min_value = float(min(values_normal));
            max_value = float(max(values_normal));
        else:
            pass;

        num_different_values = np.unique(np.vstack([values_wiederkehrer, values_normal])).shape[0];
        if num_different_values > 100:
            num_bins_hist = 100;
        else:
            num_bins_hist = num_different_values;

        print('min value: ' + str(min_value))
        print('max value: ' + str(max_value))

        range_hist = [min_value, max_value]
        # print(bins_hist)
        hist_feature_wiederkehrer, bins_wiederkehrer = np.histogram(values_wiederkehrer, range=range_hist, bins=num_bins_hist, density=True);
        hist_feature_normal, bins_normal = np.histogram(values_normal, range=range_hist, bins=num_bins_hist, density=True);
        hist_feature_wiederkehrer = hist_feature_wiederkehrer / hist_feature_wiederkehrer.sum()
        hist_feature_normal = hist_feature_normal / hist_feature_normal.sum()

        bar_width_wiederkehrer = bins_wiederkehrer[1:] - bins_wiederkehrer[:-1];
        bar_width_normal = bins_normal[1:] - bins_normal[:-1];

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 10));
        plt.bar(bins_wiederkehrer[:-1], height=hist_feature_wiederkehrer, width=bar_width_wiederkehrer, align='edge', color='b', alpha=0.5)
        plt.bar(bins_normal[:-1], height=hist_feature_normal, width=bar_width_normal, align='edge', color='m', alpha=0.5)
        plt.legend(['Wiederkehrer', 'normal'])
        plt.title(name_feature);
        plt.draw()
        plt.savefig(filename_plot, format='png')
        plt.close();

    # ideal would be to automatically select the comparison type from the feature name
    # would need to give a flag with the feature name
    # i dont know if that would be practical in the long run
    # but like this it is not ideal either
    def doFeatureComparison(self):
        df = self.dataset.getData();

        df_wiederkehrer = df['Wiederkehrer']
        print('num_wiederkehrer: ' + str(df_wiederkehrer.sum(axis=0)));

        self._doComparisonHist(df, 'ratio_los_age');
        self._doComparisonHist(df, 'ratio_numDK_age');
        self._doComparisonHist(df, 'ratio_numOE_age');
        self._doComparisonHist(df, 'ratio_los_numDK');
        self._doComparisonHist(df, 'ratio_los_numOE');
        self._doComparisonHist(df, 'mult_los_numCHOP');
        self._doComparisonHist(df, 'ratio_numCHOP_age');
        self._doComparisonHist(df, 'Eintrittsalter');
        self._doComparisonHist(df, 'Verweildauer');
        self._doComparisonHist(df, 'numDK');
        self._doComparisonHist(df, 'numOE');
        self._doComparisonHist(df, 'numCHOP');
        self._doComparisonHist(df, 'Langlieger');
        self._doComparisonHist(df, 'equalOE');
        self._doComparisonHist(df, 'previous_visits');
        self._doComparisonHist(df, 'diff_drg_alos');
        self._doComparisonHist(df, 'diff_drg_lowerbound');
        self._doComparisonHist(df, 'diff_drg_upperbound');
        self._doComparisonHist(df, 'rel_diff_drg_alos');
        self._doComparisonHist(df, 'rel_diff_drg_lowerbound');
        self._doComparisonHist(df, 'rel_diff_drg_upperbound');
        self._doComparisonHist(df, 'alos');
        self._doComparisonHist(df, 'ratio_drg_los_alos');

        self._doComparisonBar(df, 'EntlassBereich');
        self._doComparisonBar(df, 'Versicherungsklasse');
        self._doComparisonBar(df, 'Geschlecht');
        self._doComparisonBar(df, 'Forschungskonsent');
        self._doComparisonBar(df, 'Entlassjahr');
        self._doComparisonBar(df, 'Entlassmonat');
        self._doComparisonBar(df, 'Entlasstag');
        self._doComparisonBar(df, 'Aufnahmeart');
        self._doComparisonBar(df, 'Entlassart');
        self._doComparisonBar(df, 'Eintrittsart');
        self._doComparisonBar(df, 'Liegestatus');
        self._doComparisonBar(df, 'Hauptdiagnose');
        self._doComparisonBar(df, 'CHOP');

