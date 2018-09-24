
import pandas as pd

from helpers.helpers import getNumericalFeatures

from utils.Dataset import Dataset


class CategoricalDataset(Dataset):
    def __init__(self, dataset_options):
        Dataset.__init__(self, dataset_options);
        self.data = None;


    def _removeNotNeededColumns(self):
        try:
            self.data = self.data.drop('Aufnahmedatum', axis=1);
        except ValueError:
            pass;
        try:
            self.data = self.data.drop('Entlassdatum', axis=1);
        except ValueError:
            pass;
        try:
            self.data = self.data.drop('Patient', axis=1);
        except ValueError:
            pass;


    def _normalizeNumericalColumns(self):
        numerical_features = getNumericalFeatures();
        for feat in numerical_features:
            max_value = self.data[feat].max()
            min_value = self.data[feat].min()
            self.data[feat] = (self.data[feat] - min_value) / (max_value - min_value)


    def _makeDataClassifierReady(self):
        assert (self.options.getEncodingScheme() == 'categorical'), 'this function only works with the categorical encoding scheme...exit' ;

        if self.df is None:
            self._getDf();

        self.data = self.df.copy();
        self.data = self.data.fillna(0.0);
        self._removeNotNeededColumns();
        self._normalizeNumericalColumns()


    def getData(self):
        if self.data is None:
            self._makeDataClassifierReady();
        return self.data;


    def getBalancedSubset(self):
        if self.data is None:
            self._makeDataClassifierReady();

        df_pos = self.data.loc[self.data['Wiederkehrer'] == 1]
        df_neg = self.data.loc[self.data['Wiederkehrer'] == 0]
        df_pos = df_pos.sample(frac=1);
        df_neg = df_neg.sample(frac=1);

        ratio_training_samples = self.options.getRatioTrainingSamples();
        num_pos_samples = df_pos.shape[0];
        num_pos_samples_training = int(round(ratio_training_samples * num_pos_samples));
        num_pos_samples_testing = num_pos_samples - num_pos_samples_training;

        df_pos_training = df_pos.iloc[:num_pos_samples_training, :];
        df_pos_testing = df_pos.iloc[num_pos_samples_training:, :];
        df_neg_testing = df_neg.iloc[:num_pos_samples_testing, :];
        df_neg_training = df_neg.iloc[num_pos_samples_testing:, :];

        df_balanced_training = df_pos_training.append(df_neg_training);
        df_balanced_training = df_balanced_training.sample(frac=1);

        df_balanced_testing = df_pos_testing.append(df_neg_testing);
        df_balanced_testing = df_balanced_testing.sample(frac=1);
        return [df_balanced_training, df_balanced_testing];
