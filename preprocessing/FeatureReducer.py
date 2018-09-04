
import sys
import pandas as pd

from helpers.helpers import getAdminFeaturesNames

class FeatureReducer:

    def __init__(self, dir_data, dataset, filename_options_in, filename_options_out):
        self.dir_data = dir_data;
        self.dataset = dataset;
        self.filename_options_in = filename_options_in;
        self.filename_options_out = filename_options_out;
        return;


    def __getFilenameOptionsStr(self):
        if self.filename_options_in is None:
            print('filename options must not be None: ')
            print('filename_options_in: '  + str(self.filename_options_in))
        strFilenameIn = self.dataset + '_REST_' + self.filename_options_in;
        strFilenameOut = self.dataset + '_REST_' + self.filename_options_out;
        return [strFilenameIn, strFilenameOut]


    def __removeAdminFeatures(self, df):
        group_names = getAdminFeaturesNames();
        column_names = list(df.columns);

        for name in column_names:
            if name in group_names:
                df = df.drop(name, axis=1);
        print('df.shape: ' + str(df.shape));
        return df;


    def __removeAllButAdminFeatures(self, df):
        group_names = getAdminFeaturesNames();
        column_names = list(df.columns);

        for name in column_names:
            if name not in group_names:
                df = df.drop(name, axis=1);
        print('df.shape: ' + str(df.shape));
        return df;


    def reduceFeatures(self, name_reduction=None):
        print('reduction method: ' + str(name_reduction))
        if name_reduction is not None:
            [filename_str_in, filename_str_out] = self.__getFilenameOptionsStr();
            filename_data_in = self.dir_data + 'data_' + filename_str_in + '.csv';
            filename_data_out = self.dir_data + 'data_' + filename_str_out + '.csv';

            df = pd.read_csv(filename_data_in);

            if name_reduction == 'NOADMIN':
                df_reduced = self.__removeAdminFeatures(df);
            elif name_reduction == 'ONLYADMIN':
                df_reduced = self.__removeAllButAdminFeatures(df);
            else:
                print('feature reduction algorithm is not known/implemented yet...exit')
                print('possible reduction methods are: NOADMIN or ONLYADMIN');
                sys.exit();

            df_reduced.to_csv(filename_data_out, mode='w', index=False, line_terminator='\n')
        else:
            print('the reduction method needs to be named...')
