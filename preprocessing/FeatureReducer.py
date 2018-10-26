
import sys
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


    def reduceFeatures(self):
        dir_data = self.options.getDirData();
        data_prefix = self.options.getDataPrefix();

        reduction_method = self.options.getFeatureReductionSettings()['method'];
        print('reduction method: ' + str(reduction_method))
        if reduction_method is not None:
            [filename_str_in, filename_str_out] = self.__getFilenameOptionsStr();
            filename_data_in = dir_data + 'data_' + data_prefix + '_' + filename_str_in + '.csv';
            filename_data_out = dir_data + 'data_' + data_prefix + '_' + filename_str_out + '.csv';
            df = pd.read_csv(filename_data_in);

            if reduction_method == 'NOADMIN':
                df_reduced = self.__removeAdminFeatures(df);
            elif reduction_method == 'ONLYADMIN':
                df_reduced = self.__removeAllButAdminFeatures(df);
            elif reduction_method == 'NOLIEGESTATUS':
                df_reduced = self.__removeLiegestatusFeatures(df);
            elif reduction_method == 'FUSION':
                df_reduced = self.__removeAllButFusionFeatures(df);
            else:
                print('feature reduction algorithm is not known/implemented yet...exit')
                print('possible reduction methods are: NOADMIN or ONLYADMIN');
                sys.exit();

            df_reduced.to_csv(filename_data_out, mode='w', index=False, line_terminator='\n')
        else:
            print('the reduction method needs to be named...')
