
import sys

import tensorflow as tf


from utils.Dataset import Dataset


class NeuralNetDatasetHandler:

    def __init__(self, dataset_options, feature_columns, mode, balanced_datasets=True):

        self.dataset_options = dataset_options;
        self.dataset = Dataset(self.dataset_options);
        self.feature_columns = feature_columns;
        self.mode = mode;
        self.balanced_datasets = balanced_datasets;
        return;


    def _parse_csv(self, value):
        # print('Parsing', data_file)
        column_names = self.dataset.getColumnsData();
        default_values = self.feature_columns.getDefaultValues(column_names)
        columns = tf.decode_csv(value, record_defaults=default_values)
        features = dict(zip(column_names, columns))
        early_readmission_flagname = self.dataset_options.getEarlyReadmissionFlagname();
        labels = features.pop(early_readmission_flagname)
        return features, tf.equal(labels, 1)


    def _parse_csv_autoencoder(self, value):
        # print('Parsing', data_file)
        column_names = self.dataset.getColumnsData();
        default_values = self.feature_columns.getDefaultValues(column_names)
        columns = tf.decode_csv(value, record_defaults=default_values)
        features = dict(zip(column_names, columns))
        return features, columns


    def _getFilenameDatasetBalanced(self):
        filename_dataset_base = self.dataset_options.getFilename();
        if self.mode == 'train':
            filename_train = filename_dataset_base[:-4] + '_balanced_train.csv'
            filename = filename_train;
        elif self.mode == 'eval':
            filename_eval = filename_dataset_base[:-4] + '_balanced_eval.csv'
            filename = filename_eval;
        elif self.mode == 'test':
            filename_test = filename_dataset_base[:-4] + '_balanced_test.csv'
            filename = filename_test;
        else:
            print('unknown mode...exit')
            sys.exit();
        return filename;


    def _getFilenamesDatasetAll(self):
        filename_dataset_base = self.dataset_options.getFilename();
        if self.mode == 'train':
            filename_train_pos = filename_dataset_base[:-4] + '_train_pos.csv'
            filename_train_neg = filename_dataset_base[:-4] + '_train_neg.csv'
            filenames = [filename_train_pos, filename_train_neg];
        elif self.mode == 'eval':
            filename_eval_pos = filename_dataset_base[:-4] + '_eval_pos.csv'
            filename_eval_neg = filename_dataset_base[:-4] + '_eval_neg.csv'
            filenames = [filename_eval_pos, filename_eval_neg];
        elif self.mode == 'test':
            filename_test_pos = filename_dataset_base[:-4] + '_test_pos.csv'
            filename_test_neg = filename_dataset_base[:-4] + '_test_neg.csv'
            filenames = [filename_test_pos, filename_test_neg];
        else:
            print('unknown mode...exit')
            sys.exit();
        return filenames;


    def _getFilenameDatasetAutoEncoder(self):
        filename_dataset_base = self.dataset_options.getFilename();
        if self.mode == 'train':
            filename_train = filename_dataset_base[:-4] + '_balanced_train.csv'
            filename = filename_train;
        elif self.mode == 'eval':
            filename_eval = filename_dataset_base[:-4] + '_balanced_eval.csv'
            filename = filename_eval;
        elif self.mode == 'test':
            filename_test = filename_dataset_base[:-4] + '_balanced_test.csv'
            filename = filename_test;
        else:
            print('unknown mode...exit')
            sys.exit();
        return filename;


    def _dataset_reader(self):
        if self.balanced_datasets:
            filename_dataset = self._getFilenameDatasetBalanced();
            # shuffle is only performed for training; not optimal --> maybe five another flag to specify training/eval
            dataset = tf.data.TextLineDataset(filename_dataset)
            dataset = dataset.skip(1)
            if self.mode == 'train':
                dataset = dataset.shuffle(buffer_size=self.dataset.getNumSamplesBalancedSubset())
            dataset = dataset.map(self._parse_csv, num_parallel_calls=5)
            return dataset;
        else:
            filenames_dataset = self._getFilenamesDatasetAll();
            data_file_pos = filenames_dataset[0];
            data_file_neg = filenames_dataset[1];

            # Extract lines from input files using the Dataset API.
            ds_pos = tf.data.TextLineDataset(data_file_pos)
            ds_neg = tf.data.TextLineDataset(data_file_neg)

            ds_pos = ds_pos.skip(1)
            ds_neg = ds_neg.skip(1)
            ds_neg = ds_neg.map(self._parse_csv, num_parallel_calls=5)
            ds_pos = ds_pos.map(self._parse_csv, num_parallel_calls=5)

            dataset = tf.data.Dataset.zip((ds_pos, ds_neg))

            # Each input element will be converted into a two-element `Dataset` using
            # `Dataset.from_tensors()` and `Dataset.concatenate()`, then `Dataset.flat_map()`
            # will flatten the resulting `Dataset`s into a single `Dataset`.
            dataset = dataset.flat_map(
                lambda ex_pos, ex_neg: tf.data.Dataset.from_tensors(ex_pos).concatenate(
                    tf.data.Dataset.from_tensors(ex_neg)))
            if self.mode == 'train':
                dataset = dataset.shuffle(buffer_size=self.dataset.getNumSamplesBalancedSubset())
            return dataset;


    def _dataset_reader_autoencoder(self):
        if self.balanced_datasets:
            filename_dataset = self._getFilenameDatasetAutoEncoder();
            print(filename_dataset)
            # shuffle is only performed for training; not optimal --> maybe five another flag to specify training/eval
            dataset = tf.data.TextLineDataset(filename_dataset)
            dataset = dataset.skip(1)
            if self.mode == 'train':
                dataset = dataset.shuffle(buffer_size=self.dataset.getNumSamples())
            dataset = dataset.map(self._parse_csv_autoencoder, num_parallel_calls=5)
            return dataset;
        else:
            filenames_dataset = self._getFilenameDatasetAutoEncoder();
            data_file_pos = filenames_dataset[0];
            data_file_neg = filenames_dataset[1];

            # Extract lines from input files using the Dataset API.
            ds_pos = tf.data.TextLineDataset(data_file_pos)
            ds_neg = tf.data.TextLineDataset(data_file_neg)

            ds_pos = ds_pos.skip(1)
            ds_neg = ds_neg.skip(1)
            ds_neg = ds_neg.map(self._parse_csv_autoencoder, num_parallel_calls=5)
            ds_pos = ds_pos.map(self._parse_csv_autoencoder, num_parallel_calls=5)

            dataset = tf.data.Dataset.zip((ds_pos, ds_neg))

            # Each input element will be converted into a two-element `Dataset` using
            # `Dataset.from_tensors()` and `Dataset.concatenate()`, then `Dataset.flat_map()`
            # will flatten the resulting `Dataset`s into a single `Dataset`.
            dataset = dataset.flat_map(
                lambda ex_pos, ex_neg: tf.data.Dataset.from_tensors(ex_pos).concatenate(
                    tf.data.Dataset.from_tensors(ex_neg)))
            if self.mode == 'train':
                dataset = dataset.shuffle(buffer_size=self.dataset.getNumSamples())
            return dataset;


    def readDatasetTF(self):
        return self._dataset_reader();


    def readDatasetAE(self):
        return self._dataset_reader_autoencoder();



