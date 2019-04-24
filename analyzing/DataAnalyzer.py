import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd

from utils.Dataset import Dataset

class DataAnalyzer:

    def __init__(self, dataset_options, dir_plots):
        self.dataset_options = dataset_options;
        self.dataset = Dataset(dataset_options=dataset_options);
        self.dir_plots = dir_plots;
        return;


    def _printValues(self, category_names, occ_wiederkehrer, occ_normal):
        for k,name in enumerate(category_names):
            print(name + ': ' + str(occ_wiederkehrer[k]) + ' <-> ' + str(occ_normal[k]));


    def _getFeatureValues(self, df, name_feature):
        column_names = self.dataset.getColumnsDf();
        feature_columns = [];
        for col in column_names:
            if col.startswith(name_feature):
                feature_columns.append(col);
        df_feature = df[feature_columns];
        df_feature_wiederkehrer = df_feature.loc[df['Wiederkehrer'] == 1];
        df_feature_normal = df_feature.loc[df['Wiederkehrer'] == 0];
        return [df_feature_normal, df_feature_wiederkehrer];

    def _filterDFdisease(self, feature_name, feature_categories, df_feature_normal, df_feature_wiederkehrer):
        print(df_feature_wiederkehrer.shape)
        print(df_feature_normal.shape)
        series_normal = [];
        series_wiederkehrer = [];
        for cat in feature_categories:
            series_normal.append(df_feature_normal[feature_name + '_' + cat]);
            series_wiederkehrer.append(df_feature_wiederkehrer[feature_name + '_' + cat]);

        df_feature_normal_filtered = pd.concat(series_normal, axis=1);
        df_feature_wiederkehrer_filtered = pd.concat(series_wiederkehrer, axis=1);
        return [df_feature_normal_filtered, df_feature_wiederkehrer_filtered];


    # for categorical features
    def _doComparisonBar(self, df, name_feature):
        filename_plot = self.dir_plots + 'featurecomparison_' + name_feature + '.png';
        print(name_feature)
        categories_feature = self.dataset_options.getFeatureCategories(name_feature);
        if name_feature == self.dataset_options.getNameMainDiag():
            if self.dataset_options.getOptionsFiltering() in self.dataset_options.getDiseaseNames():
                categories_feature = self.dataset_options.getDiseaseICDkeys();
        print(categories_feature)
        values_to_count = range(0, len(categories_feature));

        [df_feature_normal, df_feature_wiederkehrer] = self._getFeatureValues(df, name_feature);
        if df_feature_wiederkehrer.shape[1] > 0 and df_feature_normal.shape[1] > 0:
            if name_feature == self.dataset_options.getNameMainDiag():
                if self.dataset_options.getOptionsFiltering() in self.dataset_options.getDiseaseNames():
                    [df_feature_normal, df_feature_wiederkehrer] = self._filterDFdisease(name_feature, categories_feature, df_feature_normal, df_feature_wiederkehrer);
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
        if df_feature_wiederkehrer.shape[1] > 0 and df_feature_normal.shape[1] > 0:
            num_values_normal = df_feature_normal.shape[0];
            num_values_wiederkehrer = df_feature_wiederkehrer.shape[0];
            values_wiederkehrer = df_feature_wiederkehrer.values;
            values_normal = df_feature_normal.values;

            print('normal: ' + str(df_feature_normal.shape))
            print('normal: ' + str(df_feature_wiederkehrer.shape))

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
        df = self.dataset.getDf();

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
        # self._doComparisonBar(df, 'CHOP');


    def _getRatioWiederkehrerFlag(self):
        early_readmission_flag = self.dataset_options.getEarlyReadmissionFlagname();
        df = self.dataset.getDf();
        df_wiederkehrer = df[early_readmission_flag]
        num_wiederkehrer = int(df_wiederkehrer.sum(axis=0));
        num_all = int(df.shape[0])
        print('num all: ' + str(num_all))
        print('num_wiederkehrer: ' + str(df_wiederkehrer.sum(axis=0)));
        print('ratio wiederkehrer: ' + str(float(num_wiederkehrer)/float(num_all)));


    def _getRatio18DaysReturn(self):
        df = self.dataset.getDf();
        df = df.sort_values(by=['Patient', 'Aufnahmedatum'])
        patient_ids_wiederkehrer = df['Patient'].unique();
        single_visiting_patients = 0;
        for k in range(0, len(patient_ids_wiederkehrer)):
            p_id = patient_ids_wiederkehrer[k]
            cases_df = df.loc[df['Patient'] == p_id];
            new_patient = True;
            if cases_df.shape[0] == 1:
                single_visiting_patients += 1;
            for index,row in cases_df.iterrows():
                if not new_patient:
                    timestamp_enter = row['Aufnahmedatum'];
                    diff = (datetime.fromtimestamp(timestamp_enter) - datetime.fromtimestamp(timestamp_previous_exit));
                    days = diff.days;
                    if int(days)<=18:
                        # print(str(datetime.fromtimestamp(timestamp_enter).strftime("%y,%m,%d")) + ' vs. ' + str(datetime.fromtimestamp(timestamp_previous_exit).strftime("%y,%m,%d")))
                        # print(str(int(row['Patient'])) + ': ' + ' --> ' + str(days) + ' --> ' + str(row['Wiederkehrer']))
                        df.at[index_previous,'Wiederkehrer'] = 1;
                else:
                    new_patient = False;
                timestamp_previous_exit = row['Entlassdatum'];
                index_previous = index;

        num_wiederkehrer_all = int(df['Wiederkehrer'].sum(axis=0));
        num_all = int(df.shape[0])
        print('patients with only a single visit: ' + str(single_visiting_patients))
        print('num all: ' + str(num_all))
        print('num wiedekehrer all: ' + str(num_wiederkehrer_all))
        print('ratio wiederkehrer all: ' + str(float(num_wiederkehrer_all)/float(num_all)))


    def checkWiederkehrer(self):
        self._getRatioWiederkehrerFlag();
        if self.dataset_options.getDataPrefix() == 'patrec':
            self._getRatio18DaysReturn()


    def _getNumberColumnsSubgroupPatrec(self, subgroup):
        dir_data = self.dataset_options.getDirData();
        dataset = self.dataset_options.getDatasetName();
        chunksize = self.dataset_options.getChunkSize();
        filename_data_subgroup = dir_data + 'data_patrec_' + dataset + '_' + subgroup + '_clean.csv';

        subgroup_data_reader = pd.read_csv(filename_data_subgroup, chunksize=chunksize);
        for k, chunk in enumerate(subgroup_data_reader):
            chunk = chunk.drop(self.dataset_options.getEventColumnName(), axis=1);
            columns = list(chunk.columns);
            sum_chunk = chunk.sum(axis=0);
            if k == 0:
                sum_subgroup = pd.DataFrame(data=np.zeros((1, len(columns))), columns=columns);
            sum_subgroup = sum_subgroup.add(sum_chunk);

        num_columns = int(sum_subgroup.astype(bool).sum(axis=1).values);
        print(subgroup + ' --> number of columns: ' + str(len(columns)))
        print(subgroup + ' --> number of non-zero columns: ' + str(num_columns))


    def _getAvgNumSubgroupPatrec(self, subgroup):
        dir_data = self.dataset_options.getDirData();
        dataset = self.dataset_options.getDatasetName();
        name_demographic_features = self.dataset_options.getFilenameOptionDemographicFeatures();
        encoding = self.dataset_options.getEncodingScheme();
        feature_set_str = self.dataset_options.getFeatureSetStr();
        filename_data_subgroup = dir_data + 'data_patrec_' + dataset + '_' + name_demographic_features + '_' + feature_set_str + '_' + encoding + '.csv';
        df = pd.read_csv(filename_data_subgroup);

        df_num_subgroup = df['num' + subgroup];
        avg_num = np.mean(df_num_subgroup.values);
        return avg_num;


    def _getAvgNumSubgroupNZ(self):
        dir_data = self.dataset_options.getDirData();
        dataset = self.dataset_options.getDatasetName();
        name_demographic_features = self.dataset_options.getFilenameOptionDemographicFeatures();
        grouping = self.dataset_options.getGroupingName()
        encoding = self.dataset_options.getEncodingScheme();
        feature_set_str = self.dataset_options.getFeatureSetStr();
        filename_data_subgroup = dir_data + 'data_nz_' + dataset + '_' + feature_set_str + '_' + encoding + '_' + grouping + '.csv';
        df = pd.read_csv(filename_data_subgroup);

        df_num_subgroup = df['diag_DIAG_COUNT'];
        avg_num = np.mean(df_num_subgroup.values);
        return avg_num;



    def _getNumberColumnsSubgroupNZ(self, subgroup):
        dir_data = self.dataset_options.getDirData();
        dataset = self.dataset_options.getDatasetName();
        chunksize = self.dataset_options.getChunkSize();
        filename_data_subgroup = dir_data + 'data_nz_' + dataset + '_' + subgroup + '_clean.csv';



    def _getNumberHauptdiagnosePatrec(self):
        dir_data = self.dataset_options.getDirData();
        dataset = self.dataset_options.getDatasetName();
        filename_data = dir_data + 'data_patrec_' + dataset + '_REST_clean.csv';
        df = pd.read_csv(filename_data);
        diff_values_hauptdiagnose = list(set(df['Hauptdiagnose'].values))
        print('Hauptdiagnose --> number of values: ' + str(len(diff_values_hauptdiagnose)));


    def _getNumberHauptdiagnoseNZ(self):
        dir_data = self.dataset_options.getDirData();
        dataset = self.dataset_options.getDatasetName();
        filename_data = dir_data + 'data_nz_' + dataset + '_discharge.csv';
        df = pd.read_csv(filename_data);
        diff_values_hauptdiagnose = list(set(df['main_diag'].values))
        print('Hauptdiagnose --> number of values: ' + str(len(diff_values_hauptdiagnose)));


    def getNumberColumnsSubgroup(self, subgroup):
        data_prefix = self.dataset_options.getDataPrefix();
        if data_prefix == 'patrec':
            self._getNumberColumnsSubgroupPatrec(subgroup);
        elif data_prefix == 'nz':
            pass;
        else:
            print('data prefix is unknown...exit')
            sys.exit()


    def getNumberHauptdiagnose(self):
        data_prefix = self.dataset_options.getDataPrefix();
        if data_prefix == 'patrec':
            self._getNumberHauptdiagnosePatrec();
        elif data_prefix == 'nz':
            self._getNumberHauptdiagnoseNZ()
        else:
            print('data prefix is unknown...exit')
            sys.exit();


    def getAvgNumberSubgroup(self, subgroup):
        data_prefix = self.dataset_options.getDataPrefix();
        if data_prefix == 'patrec':
            avg_num = self._getAvgNumSubgroupPatrec(subgroup);
            return avg_num;
        elif data_prefix == 'nz':
            if not subgroup == 'DK':
                print('only implemented for diagnoses...exit')
                sys.exit();
            avg_num = self._getAvgNumSubgroupNZ()
            return avg_num;
        else:
            print('unknown data prefix..exit')
            sys.exit();






