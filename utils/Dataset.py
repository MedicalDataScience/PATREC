
import pandas as pd
import numpy as np
from utils.DatasetFilter import DatasetFilter
from utils.DatasetSplitter import DatasetSplitter


class Dataset:
    
    def __init__(self, dataset_options):
        self.options = dataset_options;
        self.df = None;
        self.columns_df = None;
        self.data = None;
        self.columns_data = None;
        return;


    # maybe stop creating separate files for filtered datasets and just create the df on the fly
    def _filterData(self):
        diseases = self.options.getDiseaseNames();
        filter = DatasetFilter(self.options)
        options_filtering = self.options.getOptionsFiltering();
        if options_filtering in diseases:
            self.df = filter.filterDataDisease()
        elif options_filtering.split("_")[0] in self.options.getCategoricalFeatures() and not self.options.getEncodingScheme() == 'categorical':
            self.df = filter.filterCategoricalColumn(options_filtering)
        else:
            self.df = filter.filterDataBinaryColumns(options_filtering)


    def _getDf(self):
        if self.options.getOptionsFiltering() is not None:
            self._filterData();
        else:
            filename = self.options.getFilename()
            df = pd.read_csv(filename);
            self.df = df;
    

    def _getColumnsDf(self):
        cols = list(self.df.columns);
        self.columns_df = cols;

    def _getColumnsData(self):
        if self.data is None:
            self._getData();
        cols = list(self.data.columns);
        self.columns_data = cols;


    def _removeNotNeededColumns(self):
        not_needed_columns = self.options.getColumnsToRemove();
        columns_data = list(self.data.columns);
        for col in not_needed_columns:
            if col in columns_data:
                try:
                    self.data = self.data.drop(col, axis=1);
                except ValueError or KeyError:
                    pass;


    def _normalizeNumericalColumns(self):
        if self.columns_data is None:
            self._getColumnsData();
        for feat in self.columns_data:
            max_value = self.data[feat].max()
            min_value = self.data[feat].min()
            if not max_value == min_value:
                self.data[feat] = (self.data[feat] - min_value) / (max_value - min_value)


    def _getData(self):
        if self.df is None:
            self._getDf();
        self.data = self.df.copy();
        self.data = self.data.fillna(0.0);
        self._removeNotNeededColumns();
        if self.options.getEncodingScheme() == 'categorical':
            self._normalizeNumericalColumns();


    def _splitData(self):
        if self.data is None:
            self.getData();

        early_readmission_flagname = self.options.getEarlyReadmissionFlagname();
        df_pos = self.data.loc[self.data[early_readmission_flagname] == 1]
        df_neg = self.data.loc[self.data[early_readmission_flagname] == 0]
        df_pos = df_pos.sample(frac=1);
        df_neg = df_neg.sample(frac=1);
        return [df_pos, df_neg];


    def _getBalancedSubset(self):
        [df_pos, df_neg] = self._splitData();
        num_pos_samples = df_pos.shape[0];
        num_neg_samples = df_neg.shape[0];
        min_num_samples = int(np.min([num_pos_samples, num_neg_samples]));
        df_pos_balanced = df_pos[:min_num_samples];
        df_neg_balanced = df_neg[:min_num_samples];
        return [df_pos_balanced, df_neg_balanced];


    def _getTrainingTesting(self):
        ratio_training_samples = self.options.getRatioTrainingSamples();

        [df_pos, df_neg] = self._splitData();
        num_pos_samples = df_pos.shape[0];
        num_pos_samples_training = int(round(ratio_training_samples * num_pos_samples));
        num_pos_samples_testing = num_pos_samples - num_pos_samples_training;

        df_pos_training = df_pos.iloc[:num_pos_samples_training, :];
        df_pos_testing = df_pos.iloc[num_pos_samples_training:, :];
        print('df_pos_training: ' + str(df_pos_training.shape))
        print('df_pos_testing: ' + str(df_pos_testing.shape))
        df_neg_testing = df_neg.iloc[:num_pos_samples_testing, :];
        df_neg_training = df_neg.iloc[num_pos_samples_testing:, :];
        print('df_neg_training: ' + str(df_neg_training.shape))
        print('df_neg_testing: ' + str(df_neg_testing.shape))
        training = [df_pos_training, df_neg_training];
        testing = [df_pos_testing, df_neg_testing];
        return [training, testing]


    def getColumnsDf(self):
        if self.df is None:
            self._getDf();
        if self.columns_df is None:
            self._getColumnsDf();
        return self.columns_df;


    def getColumnsData(self):
        if self.data is None:
            self._getData();
        if self.columns_data is None:
            self._getColumnsData();
        return self.columns_data;


    def getDf(self):
        if self.df is None:
            self._getDf();
        return self.df;


    def getData(self):
        if self.data is None:
            self._getData();
        return self.data;


    def getFilename(self, filteroptions=False):
        return self.options.getFilename(filteroptions);


    def getFilenameOptions(self, filteroptions=False):
        return self.options.getFilenameOptions(filteroptions);


    def getBalancedSubsetTrainingAndTesting(self):
        [df_pos, df_neg] = self._getBalancedSubset();
        ratio_training_samples = self.options.getRatioTrainingSamples();
        num_pos_samples = df_pos.shape[0];
        num_pos_samples_training = int(round(ratio_training_samples * num_pos_samples));
        num_pos_samples_testing = num_pos_samples - num_pos_samples_training;

        df_pos_training = df_pos.iloc[:num_pos_samples_training, :];
        df_neg_training = df_neg.iloc[:num_pos_samples_training, :];
        df_pos_testing = df_pos.iloc[-num_pos_samples_testing:, :];
        df_neg_testing = df_neg.iloc[-num_pos_samples_testing:, :];

        df_balanced_training = df_pos_training.append(df_neg_training);
        df_balanced_training = df_balanced_training.sample(frac=1);
        df_balanced_testing = df_pos_testing.append(df_neg_testing);
        df_balanced_testing = df_balanced_testing.sample(frac=1);

        return [df_balanced_training, df_balanced_testing];


    def getTrainingAndTestingSet(self):
        [training, testing] = self._getTrainingTesting();
        return [training, testing]


    def getBalancedSubSet(self):
        [df_pos, df_neg] = self._getBalancedSubset();
        df_balanced = df_pos.append(df_neg);
        df_balanced = df_balanced.sample(frac=1);
        return df_balanced;


    def splitDatasetIntoTrainingTestingSet(self):
        datasplitter = DatasetSplitter(self.options)
        datasplitter.splitDatasetIntoTrainingTesting();


    def getNumSamplesBalancedSubset(self):
        [df_pos, df_neg] = self._getBalancedSubset();
        df_balanced = df_pos.append(df_neg);
        num_samples = df_balanced.shape[0];
        return num_samples;

    def getNumSamples(self):
        if self.df is None:
            self._getDf();
        num_samples = self.df.shape[0];
        return num_samples;

