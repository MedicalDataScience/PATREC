import os
import sys
from datetime import datetime

import pandas as pd


class FeatureReducer:

    def __init__(self, options_dataset, filename_options_in):
        self.options = options_dataset;
        self.filename_options_in = filename_options_in;
        return;


    def __getFilenameOptionsStr(self):
        dataset = self.options.getDatasetName();
        name_dem_features = self.options.getFilenameOptionDemographicFeatures();
        filename_options_out = self.options.getFeatureSetStr();

        strFilenameIn = dataset + '_' + name_dem_features;
        if self.filename_options_in is not None:
            strFilenameIn = strFilenameIn + '_' + self.filename_options_in;
        strFilenameOut = dataset + '_' + name_dem_features + '_' + filename_options_out;
        return [strFilenameIn, strFilenameOut]


    def __removeAdminFeatures(self, df):
        group_names = self.options.getAdminFeatureNames();
        column_names = list(df.columns);

        for name in column_names:
            if name in group_names:
                df = df.drop(name, axis=1);
        print('df.shape: ' + str(df.shape));
        return df;


    def __removeAllButAdminFeatures(self, df):
        group_names = self.options.getAdminFeatureNames();
        column_names = list(df.columns);

        for name in column_names:
            if name not in group_names:
                df = df.drop(name, axis=1);
        print('df.shape: ' + str(df.shape));
        return df;


    def __removeLiegestatusFeatures(self,df):
        group_names = self.options.getLiegestatusFeatureNames();
        column_names = list(df.columns);
        for name in column_names:
            if name in group_names:
                df = df.drop(name, axis=1);
        print('df.shape: ' + str(df.shape))
        return df;


    def __removeAllButFusionFeatures(self, df):
        fusion_features = self.options.getFusionFeatureNames();
        early_readmission_flagname = self.options.getEarlyReadmissionFlagname();
        column_names = list(df.columns);
        for name in column_names:
            if name == early_readmission_flagname:
                continue;
            elif name in fusion_features:
                pass;
            else:
                df = df.drop(name, axis=1);
        print('df.shape: ' + str(df.shape));
        return df;


    def _changeWiederkehrerDef(self, df):
        df = df.sort_values(by=['Patient', 'Aufnahmedatum'])
        patient_ids_wiederkehrer = df['Patient'].unique();
        single_visiting_patients = 0;
        for k in range(0, len(patient_ids_wiederkehrer)):
            p_id = patient_ids_wiederkehrer[k]
            cases_df = df.loc[df['Patient'] == p_id];
            new_patient = True;
            if cases_df.shape[0] == 1:
                single_visiting_patients += 1;
            for index, row in cases_df.iterrows():
                if not new_patient:
                    timestamp_enter = row['Aufnahmedatum'];
                    diff = (datetime.fromtimestamp(timestamp_enter) - datetime.fromtimestamp(timestamp_previous_exit));
                    days = diff.days;
                    if int(days) <= 18:
                        # print(str(datetime.fromtimestamp(timestamp_enter).strftime("%y,%m,%d")) + ' vs. ' + str(datetime.fromtimestamp(timestamp_previous_exit).strftime("%y,%m,%d")))
                        # print(str(int(row['Patient'])) + ': ' + ' --> ' + str(days) + ' --> ' + str(row['Wiederkehrer']))
                        df.at[index_previous, 'Wiederkehrer'] = 1;
                else:
                    new_patient = False;
                timestamp_previous_exit = row['Entlassdatum'];
                index_previous = index;
        return df;

    def reduceFeatures(self):
        dir_data = self.options.getDirData();
        data_prefix = self.options.getDataPrefix();

        reduction_method = self.options.getFeatureReductionSettings()['method'];
        print('reduction method: ' + str(reduction_method))
        if reduction_method is not None:
            [filename_str_in, filename_str_out] = self.__getFilenameOptionsStr();
            filename_data_in = os.path.join(dir_data, data_prefix + '_' + filename_str_in + '.csv');
            filename_data_out = os.path.join(dir_data, data_prefix + '_' + filename_str_out + '.csv');
            df = pd.read_csv(filename_data_in);

            if reduction_method == 'NOADMIN':
                df_reduced = self.__removeAdminFeatures(df);
            elif reduction_method == 'ONLYADMIN':
                df_reduced = self.__removeAllButAdminFeatures(df);
            elif reduction_method == 'NOLIEGESTATUS':
                df_reduced = self.__removeLiegestatusFeatures(df);
            elif reduction_method == 'FUSION':
                if data_prefix == 'patrec':
                    df = self._changeWiederkehrerDef(df);
                df_reduced = self.__removeAllButFusionFeatures(df);
            elif reduction_method == 'ONLYDIAG':
                pass
            else:
                print('feature reduction algorithm is not known/implemented yet...exit')
                print('possible reduction methods are: NOADMIN or ONLYADMIN');
                sys.exit();
            if not reduction_method == 'ONLYDIAG':
                df_reduced.to_csv(filename_data_out, mode='w', index=False, line_terminator='\n')
        else:
            print('the reduction method needs to be named...')
