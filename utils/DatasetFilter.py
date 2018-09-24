
import pandas as pd

class DatasetFilter:

    def __init__(self, options_dataset):
        self.options = options_dataset;
        return;


    def __filterBinaryColumn(self, df, key):
        print('df.shape: ' + str(df.shape))
        df_key = df[key];
        df_filtered = df.loc[df_key == 1];
        print('df_filtered: ' + str(df_filtered.shape));
        return df_filtered;


    def filterDataBinaryColumns(self, filterKey):
        dir_data = self.options.getDirData();
        dataset = self.options.getDatasetName();
        featureset = self.options.getFeatureSet();
        encoding = self.options.getEncodingScheme();
        grouping = self.options.getGroupingName();
        filename_options_in = featureset + '_' + encoding + '_' + grouping;

        strFilenameIn = dataset + '_' + filename_options_in;
        strFilenameOut = strFilenameIn + '_' + filterKey;
        filename_data_in = dir_data + 'data_' + strFilenameIn + '.csv';
        filename_data_out = dir_data + 'data_' + strFilenameOut + '.csv';

        df = pd.read_csv(filename_data_in);
        df_filtered = self.__filterBinaryColumn(df, filterKey);
        df_filtered.to_csv(filename_data_out, line_terminator='\n', index=False);
