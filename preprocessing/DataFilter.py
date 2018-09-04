
import pandas as pd

class DataFilter:

    def __init__(self, dir_data, dataset, filename_options_in):
        self.dir_data = dir_data;
        self.dataset = dataset;
        self.filename_options_in = filename_options_in;
        return;


    def __filterBinaryColumn(self, df, key):
        print('df.shape: ' + str(df.shape))
        df_key = df[key];
        df_filtered = df.loc[df_key == 1];
        print('df_filtered: ' + str(df_filtered.shape));
        return df_filtered;


    def filterDataBinaryColumns(self, filterKey):
        strFilenameIn = self.dataset + '_' + self.filename_options_in;
        strFilenameOut = strFilenameIn + '_' + filterKey;
        filename_data_in = self.dir_data + 'data_' + strFilenameIn + '.csv';
        filename_data_out = self.dir_data + 'data_' + strFilenameOut + '.csv';

        df = pd.read_csv(filename_data_in);
        df_filtered = self.__filterBinaryColumn(df, filterKey);
        df_filtered.to_csv(filename_data_out, line_terminator='\n', index=False);
