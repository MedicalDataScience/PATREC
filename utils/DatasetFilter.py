import sys
import pandas as pd

class DatasetFilter:

    def __init__(self, options_dataset):
        self.options = options_dataset;
        return;


    def _filterColumnsDiagnosisDiseases(self, df):
        keys = self.options.getDiseaseICDkeys();
        df_disease = pd.DataFrame(columns=list(df.columns));
        for key in keys:
            df_key = df[self.options.getNameMainDiag() + '_' + key];
            df_disease = pd.concat([df_disease, df.loc[df_key == 1]], axis=0)
        print('df_disease: ' + str(df_disease.shape))
        return df_disease;



    def __filterBinaryColumn(self, df, key):
        print('df.shape: ' + str(df.shape))
        df_key = df[key];
        df_filtered = df.loc[df_key == 1];
        print('df_filtered: ' + str(df_filtered.shape));
        return df_filtered;


    def filterDataBinaryColumns(self, filterKey):
        dir_data = self.options.getDirData();
        data_prefix = self.options.getDataPrefix();
        dataset = self.options.getDatasetName();
        featureset_str = self.options.getFeatureSetStr();
        encoding = self.options.getEncodingScheme();
        grouping = self.options.getGroupingName();
        filename_options_in = featureset_str + '_' + encoding + '_' + grouping;

        strFilenameIn = dataset + '_' + filename_options_in;
        strFilenameOut = strFilenameIn + '_' + filterKey;
        filename_data_in = dir_data + 'data_' + data_prefix + '_' + strFilenameIn + '.csv';
        # filename_data_out = dir_data + 'data_' + strFilenameOut + '.csv';

        df = pd.read_csv(filename_data_in);
        df_filtered = self.__filterBinaryColumn(df, filterKey);
        # df_filtered.to_csv(filename_data_out, line_terminator='\n', index=False);
        return df_filtered;



    def filterDataDisease(self):
        dir_data = self.options.getDirData();
        data_prefix = self.options.getDataPrefix();
        dataset = self.options.getDatasetName();
        featureset_str = self.options.getFeatureSetStr();
        encoding = self.options.getEncodingScheme();
        grouping = self.options.getGroupingName();
        disease_name = self.options.getOptionsFiltering();
        filename_options_in = featureset_str + '_' + encoding + '_' + grouping;

        strFilenameIn = dataset + '_' + filename_options_in;
        strFilenameOut = strFilenameIn + '_' + disease_name;
        filename_data_in = dir_data + 'data_' + data_prefix + '_' + strFilenameIn + '.csv';
        filename_data_out = dir_data + 'data_' + data_prefix + '_' + strFilenameOut + '.csv';

        df = pd.read_csv(filename_data_in);
        df_filtered = self._filterColumnsDiagnosisDiseases(df);
        df_filtered.to_csv(filename_data_out, line_terminator='\n', index=False);
        return df_filtered;