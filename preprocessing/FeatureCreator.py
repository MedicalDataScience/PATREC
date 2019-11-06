import os
import sys
import sys
import pandas as pd



class FeatureCreator:

    def __init__(self, options_dataset, filename_options_in, filename_options_out):
        self.options = options_dataset;
        self.filename_options_in = filename_options_in;
        self.filename_options_out = filename_options_out;
        self.additional_features = self.options.getNewFeatureSettings()['names'];
        return;


    def __getFilenameOptionStr(self):
        dataset = self.options.getDatasetName();
        name_dem_features = self.options.getFilenameOptionDemographicFeatures();

        if self.filename_options_in is None or self.filename_options_out is None:
            print('filename options must not be None: ')
            print('filename_options_in: '  + str(self.filename_options_in))
            print('filename_options_out: ' + str(self.filename_options_out))

        strFilenameIn = dataset + '_' + name_dem_features + '_' + self.filename_options_in;
        strFilenameOut = dataset + '_' + name_dem_features + '_' + self.filename_options_out;
        return [strFilenameIn, strFilenameOut]


    def __calculatePreviousVisits(self, df):
        df_sorted = df.sort_values(by=['Patient', 'Aufnahmedatum']);
        patient_ids = df_sorted['Patient'].unique();
        df['previous_visits'] = 0;
        for p_id in patient_ids:
            df_patient = df_sorted.loc[df_sorted['Patient'] == p_id];
            cnt_visits = 0;
            for k,row in df_patient.iterrows():
                df.at[k,'previous_visits'] = cnt_visits;
                cnt_visits += 1;
        return df['previous_visits'];

    def __calculateDRGDiffYear(self, df, df_drg, feature_name, year):
        dir_data = self.options.getDirData();
        filename_swissdrg_year = os.path.join(dir_data, 'swissdrg_alos_' + str(year) + '.xlsx');
        df_drg_year = pd.read_excel(filename_swissdrg_year, sheetname='Akutspitäler');

        cnt_drgnotfound = 0;
        for k,row in df_drg.iterrows():
            drgcode = row['DRGCode'];
            los = row['Verweildauer'];
            feature_val = df_drg_year.loc[df_drg_year['DRGCode'] == drgcode][feature_name].values;
            try:
                diff = los - feature_val;
                df.at[k, 'diff_drg_' + feature_name] = float(diff);
            except TypeError:
                cnt_drgnotfound += 1;
        print('number of not-found drg codes: ' + str(cnt_drgnotfound))
        return df;


    def __calculateDRGDiff(self, df, feature_name):
        name_newfeature = 'diff_drg_' + feature_name;
        drg_years = [2012, 2013, 2014, 2015, 2016, 2017, 2018];
        df[name_newfeature] = 0.0;
        for y in drg_years:
            df_drg = df.loc[df[name_newfeature] == 0.0];
            df = self.__calculateDRGDiffYear(df, df_drg, feature_name, y);
            df_drg_updated = df.loc[df[name_newfeature] == 0.0];
            print('df_drg_updated: ' + str(df_drg_updated.shape))
        return df[name_newfeature];


    def __calculateRelDRGDiffYear(self, df, df_drg, feature_name, year):
        dir_data = self.options.getDirData();
        filename_swissdrg_year = os.path.join(dir_data, 'swissdrg_alos_' + str(year) + '.xlsx');
        df_drg_year = pd.read_excel(filename_swissdrg_year, sheetname='Akutspitäler');

        name_newfeature = 'rel_diff_drg_' + feature_name
        cnt_drgnotfound = 0;
        for k, row in df_drg.iterrows():
            drgcode = row['DRGCode'];
            los = float(row['Verweildauer']);
            try:
                feature_val = float(df_drg_year.loc[df_drg_year['DRGCode'] == drgcode][feature_name].values);
                diff = los - feature_val;
                rel_diff = diff / float(feature_val);
                df.at[k, name_newfeature] = float(rel_diff);
            except TypeError:
                cnt_drgnotfound += 1;
        print('number of not-found drg codes: ' + str(cnt_drgnotfound))
        return df;


    def __calculateRelDRGDiff(self, df, feature_name):
        name_newfeature = 'rel_diff_drg_' + feature_name;
        drg_years = [2012, 2013, 2014, 2015, 2016, 2017, 2018];
        df[name_newfeature] = 0.0;
        for y in drg_years:
            df_drg = df.loc[df[name_newfeature] == 0.0];
            df = self.__calculateRelDRGDiffYear(df, df_drg, feature_name, y);
            df_drg_updated = df.loc[df[name_newfeature] == 0.0];
            print('df_drg_updated: ' + str(df_drg_updated.shape))
        return df[name_newfeature];


    def __calculateDRGalosYear(self, df, df_drg, feature_name, year):
        dir_data = self.options.getDirData();
        filename_swissdrg_year = os.path.join(dir_data, 'swissdrg_alos_' + str(year) + '.xlsx');
        df_drg_year = pd.read_excel(filename_swissdrg_year, sheetname='Akutspitäler');
        cnt_drgnotfound = 0;
        for k, row in df_drg.iterrows():
            drgcode = row['DRGCode'];
            feature_val = df_drg_year.loc[df_drg_year['DRGCode'] == drgcode][feature_name].values;
            try:
                df.at[k, feature_name] = float(feature_val);
            except TypeError:
                cnt_drgnotfound += 1;
        print('number of not-found drg codes: ' + str(cnt_drgnotfound))
        return df;


    def __calculateDRGalos(self, df, feature_name):
        drg_years = [2012, 2013, 2014, 2015, 2016, 2017, 2018];
        df[feature_name] = 0.0;
        for y in drg_years:
            df_drg = df.loc[df[feature_name] == 0.0];
            df = self.__calculateDRGalosYear(df, df_drg, feature_name, y);
            df_drg_updated = df.loc[df[feature_name] == 0.0];
            print('df_drg_updated: ' + str(df_drg_updated.shape))
        return df[feature_name];


    def __calculateDRGRatioLosYear(self, df, df_drg, feature_name, year):
        dir_data = self.options.getDirData();
        filename_swissdrg_year = os.path.join(dir_data, 'swissdrg_alos_' + str(year) + '.xlsx');
        df_drg_year = pd.read_excel(filename_swissdrg_year, sheetname='Akutspitäler');
        name_newfeature = 'ratio_drg_los_' + feature_name;
        cnt_drgnotfound = 0;
        for k, row in df_drg.iterrows():
            drgcode = row['DRGCode'];
            feature_val = df_drg_year.loc[df_drg_year['DRGCode'] == drgcode][feature_name].values;
            los = float(row['Verweildauer']);
            try:
                ratio = los / float(feature_val);
                df.at[k, name_newfeature] = float(ratio);
            except TypeError:
                cnt_drgnotfound += 1;
        print('number of not-found drg codes: ' + str(cnt_drgnotfound))
        return df;


    def __calculateDRGRatioLos(self, df, feature_name):
        drg_years = [2012, 2013, 2014, 2015, 2016, 2017, 2018];
        name_newfeature = 'ratio_drg_los_' + feature_name;
        df[name_newfeature] = 0.0;
        for y in drg_years:
            df_drg = df.loc[df[name_newfeature] == 0.0];
            df = self.__calculateDRGRatioLosYear(df, df_drg, feature_name, y);
            df_drg_updated = df.loc[df[name_newfeature] == 0.0];
            print('df_drg_updated: ' + str(df_drg_updated.shape))
        return df[name_newfeature];


    def __generateAdditionalFeatures(self, df):
        for newfeature in self.additional_features:
            print('')
            print(newfeature)
            if newfeature == 'ratio_los_age':
                df['ratio_los_age'] = df['Verweildauer'] / df['Eintrittsalter'];
            elif newfeature == 'ratio_numDK_age':
                df['ratio_numDK_age'] = df['numDK'] / df['Eintrittsalter'];
            elif newfeature == 'ratio_los_numDK':
                df['ratio_los_numDK'] = df['Verweildauer'] / df['numDK'];
            elif newfeature == 'ratio_numCHOP_age':
                df['ratio_numCHOP_age'] = df['numCHOP'] / df['Eintrittsalter'];
            elif newfeature == 'ratio_los_numOE':
                df['ratio_los_numOE'] = df['Verweildauer'] / df['numOE'];
            elif newfeature == 'ratio_numOE_age':
                df['ratio_numOE_age'] = df['numOE'] / df['Eintrittsalter'];
            elif newfeature == 'mult_los_numCHOP':
                df['mult_los_numCHOP'] = df['Verweildauer'] * df['numCHOP'];
            elif newfeature == 'mult_equalOE_numDK':
                df['mult_equalOE_numDK'] = df['equalOE'] * df['numDK'];
            elif newfeature == 'previous_visits':
                df[newfeature] = self.__calculatePreviousVisits(df);
            elif newfeature == 'diff_drg_alos':
                df[newfeature] = self.__calculateDRGDiff(df, 'alos');
            elif newfeature == 'diff_drg_lowerbound':
                df[newfeature] = self.__calculateDRGDiff(df, 'lowerbound');
            elif newfeature == 'diff_drg_upperbound':
                df[newfeature] = self.__calculateDRGDiff(df, 'upperbound');
            elif newfeature == 'rel_diff_drg_alos':
                df[newfeature] = self.__calculateRelDRGDiff(df, 'alos');
            elif newfeature == 'rel_diff_drg_lowerbound':
                df[newfeature] = self.__calculateRelDRGDiff(df, 'lowerbound');
            elif newfeature == 'rel_diff_drg_upperbound':
                df[newfeature] = self.__calculateRelDRGDiff(df, 'upperbound');
            elif newfeature == 'alos':
                df[newfeature] = self.__calculateDRGalos(df, 'alos');
            elif newfeature == 'ratio_drg_los_alos':
                df[newfeature] = self.__calculateDRGRatioLos(df, 'alos');
            else:
                print('this additional feature is not yet implemented...sorry')
                sys.exit()
        return df;


    def __createFeatureNumOccurrences(self, name_subgroup):
        chunksize = self.options.getChunkSize();
        dir_data = self.options.getDirData();
        dataset = self.options.getDatasetName();
        data_prefix = self.options.getDataPrefix();
        df_num = pd.DataFrame(columns=['num' + name_subgroup]);
        strFilenameIn = dataset + '_' + name_subgroup + '_clean';
        filename_data_subgroup_in = os.path.join(dir_data, data_prefix + '_' + strFilenameIn + '.csv');
        data_reader = pd.read_csv(filename_data_subgroup_in, chunksize=chunksize);
        for k, chunk in enumerate(data_reader):
            chunk = chunk.drop('Fall', axis=1);
            chunk_new = pd.DataFrame(index=chunk.index, columns=['num'+name_subgroup]);
            chunk_new['num'+name_subgroup] = chunk.sum(axis=1)
            df_num = df_num.append(chunk_new)
        print('df_num: ' + str(df_num.shape))
        return df_num;


    def __addFeaturesFromCleanData(self, df):
        subgroups = self.options.getSubgroups();
        #create cnt features: numCHOP, numDK, numOE
        for sub in subgroups:
            feature_name = 'num' + sub;
            df_newfeature = self.__createFeatureNumOccurrences(sub);
            df[feature_name] = df_newfeature.values;
            print('df.shape: ' + str(df.shape))
            if sub == 'OE':
                df_newfeature = pd.DataFrame(index=df.index, columns=['equalOE']);
                df_newfeature = df_newfeature.fillna(0);
                df_newfeature['equalOE'] = (df['EntlassOE'] == df['AufnehmOE']).astype(int);
                df['equalOE'] = df_newfeature.values;
        print('df.shape: ' + str(df.shape))
        return df;


    def addFeatures(self):
        assert self.additional_features is not None, 'names new features cannot be None when choosing this featureset...exit'
        if len(self.additional_features) == 0:
            print('WARNING: empty list of names for new features...');

        dir_data = self.options.getDirData();
        data_prefix = self.options.getDataPrefix();
        [filename_str_in, filename_str_out] = self.__getFilenameOptionStr()
        filename_data_in = os.path.join(dir_data, data_prefix + '_' + filename_str_in + '.csv');
        filename_data_out = os.path.join(dir_data, data_prefix + '_' + filename_str_out + '.csv');
        df = pd.read_csv(filename_data_in);

        df_newfeatures = self.__addFeaturesFromCleanData(df);
        df_additional_features = self.__generateAdditionalFeatures(df_newfeatures);
        df_newfeatures = df_additional_features;

        df_newfeatures.to_csv(filename_data_out, line_terminator='\n', index=False);