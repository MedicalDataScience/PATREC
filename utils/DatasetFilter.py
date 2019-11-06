import sys, os
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

    def _filterColumnsDiagnosisDiseasesEmbedding(self, df):
        keys = self.options.getDiseaseICDkeys();
        df_disease = pd.DataFrame(columns=list(df.columns));
        for key in keys:
            df_key = df.loc[df[self.options.getNameMainDiag()] == key];
            df_disease = pd.concat([df_disease, df_key], axis=0)
        print('df_disease: ' + str(df_disease.shape))
        return df_disease;

    def __filterBinaryColumn(self, df, key):
        print('df.shape: ' + str(df.shape))
        df_key = df[key];
        df_filtered = df.loc[df_key == 1];
        print('df_filtered: ' + str(df_filtered.shape));
        return df_filtered;

    def _filterCategoricalColumn(self, df, key):
        col_name, key = key.split("_")
        df_filtered = df.loc[df[col_name] == key]
        return df_filtered


    def filterDataBinaryColumns(self, filterKey):
        filename_data_in = self.options.getFilename();
        df = pd.read_csv(filename_data_in);
        df_filtered = self.__filterBinaryColumn(df, filterKey);
        return df_filtered;


    def filterCategoricalColumn(self, filterKey):
        filename_data_in = self.options.getFilename();
        df = pd.read_csv(filename_data_in);
        df_filtered = self._filterCategoricalColumn(df, filterKey);
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
        filename_data_in = os.path.join(dir_data, 'data_' + data_prefix + '_' + strFilenameIn + '.csv');
        filename_data_out = os.path.join(dir_data, 'data_' + data_prefix + '_' + strFilenameOut + '.csv');

        df = pd.read_csv(filename_data_in);
        if encoding == 'embedding':
            df_filtered = self._filterColumnsDiagnosisDiseasesEmbedding(df);
        else:
            df_filtered = self._filterColumnsDiagnosisDiseases(df);
        df_filtered.to_csv(filename_data_out, line_terminator='\n', index=False);
        return df_filtered;


    def filterDataEntlassBereich(self):
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
        filename_data_in = os.path.join(dir_data, 'data_' + data_prefix + '_' + strFilenameIn + '.csv');
        filename_data_out = os.path.join(dir_data, 'data_' + data_prefix + '_' + strFilenameOut + '.csv');

        df = pd.read_csv(filename_data_in);
        df_filtered = self._filterColumnsDiagnosisDiseases(df);
        df_filtered.to_csv(filename_data_out, line_terminator='\n', index=False);
        return df_filtered;