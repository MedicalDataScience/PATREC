import os
import sys
import shutil
import numpy as np

import tensorflow as tf
from tensorflow.python.summary import summary

from learning.neuralnet.AutoEncoderEstimator import AutoEncoderEstimator
from learning.neuralnet.NeuralNetDatasetHandler import NeuralNetDatasetHandler
from learning.neuralnet.NeuralNetDatasetMaker import NeuralNetDatasetMaker
from utils.Dataset import Dataset

from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers

class AutoEncoderModel():

    def __init__(self, mode, dict_dataset_options, feature_columns, flags):
        self.feature_columns = feature_columns;
        self.dataset_options_train = dict_dataset_options['train'];
        self.dataset_options_eval = dict_dataset_options['eval'];
        self.dataset_options_test = dict_dataset_options['test'];
        self.mode = mode;

        self.flags = flags;
        self.model = None;
        self.flags.hidden_units = [int(u) for u in self.flags.hidden_units];

        if self.mode == 'train':
            self._setModelDir();
            # Clean up the model directory if present
            if not self.flags.model_dir == self.flags.pretrained_model_dir:
                shutil.rmtree(self.flags.model_dir, ignore_errors=True)
            if not os.path.exists(self.flags.model_dir):
                os.makedirs(self.flags.model_dir);

        if self.mode == 'train':
            if not self.dataset_options_eval is None:
                self.dataset_handler_train = NeuralNetDatasetHandler(self.flags.model_dir, self.dataset_options_train, feature_columns, 'train');
                self.dataset_handler_eval = NeuralNetDatasetHandler(self.flags.model_dir, self.dataset_options_eval, feature_columns, 'eval');
            else:
                self.dataset_handler_train = NeuralNetDatasetHandler(self.flags.model_dir, self.dataset_options_train, feature_columns, 'train');
                self.dataset_handler_eval = NeuralNetDatasetHandler(self.flags.model_dir, self.dataset_options_train, feature_columns, 'eval');
        elif self.mode == 'test':
            self.dataset_handler_test = NeuralNetDatasetHandler(self.flags.model_dir, self.dataset_options_test, feature_columns, 'test');

        return;


    def _setModelDir(self):
        modeldir_base = self.flags.model_dir;
        dropoutrate = self.flags.dropout;
        learningrate = self.flags.learningrate;
        batchnorm = self.flags.batchnorm;
        batchsize = self.flags.batch_size;
        dataset_filename_options = self.dataset_options_train.getFilenameOptions(filteroptions=False);
        suffix_modeldir = 'autoencoder_';
        if self.flags.continue_training:
            suffix_modeldir = 'autoencoder_warmstart_';
        suffix_modeldir = suffix_modeldir + dataset_filename_options + '_' + str(self.flags.hidden_units[0]);

        for k in range(1, len(self.flags.hidden_units)):
            suffix_modeldir = suffix_modeldir + '_' + str(self.flags.hidden_units[k]);

        suffix_modeldir = suffix_modeldir + '_dropout_' + str(dropoutrate);
        suffix_modeldir = suffix_modeldir + '_learningrate_' + str(learningrate);
        suffix_modeldir = suffix_modeldir + '_batchnorm_' + str(batchnorm);
        suffix_modeldir = suffix_modeldir + '_batchsize_' + str(batchsize);
        model_dir = modeldir_base + '/' + suffix_modeldir;
        self.flags.model_dir = model_dir;


    def _input_fn_train(self):
        dataset = self.dataset_handler_train.readDatasetAE();
        dataset = dataset.repeat(self.flags.epochs_between_evals);
        dataset = dataset.batch(self.flags.batch_size);
        return dataset;

    def _input_fn_analyze(self):
        diag_group_names = self.dataset_options_train.getDiagGroupNames();
        features = {'diag': diag_group_names};
        dataset = tf.data.Dataset.from_tensor_slices((features, features))
        dataset = dataset.repeat(1);
        dataset = dataset.batch(self.flags.batch_size);
        return dataset;


    def _input_fn_encode_diag(self):
        dataset = self.dataset_handler_test.readDatasetAE();
        dataset = dataset.repeat(self.flags.epochs_between_evals);
        dataset = dataset.batch(self.flags.batch_size);
        return dataset;

    def _input_fn_encode_maindiag(self):
        dataset = self.dataset_handler_test.getDatasetEncodeMainDiag();
        dataset = dataset.repeat(self.flags.epochs_between_evals);
        dataset = dataset.batch(self.flags.batch_size);
        return dataset;



    def export_model(self):
        """Export to SavedModel format.
        Args:
        model: Estimator object
        export_dir: directory to export the model.
        """
        estimator = self.model.getEstimator();
        deep_columns = self.feature_columns.buildModelColumns()
        feature_spec = tf.feature_column.make_parse_example_spec(deep_columns)
        example_input_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
        r = estimator.export_savedmodel(self.flags.export_dir, example_input_fn)


    def _getModelEstimator(self):
        if self.model is None:
            self.model = AutoEncoderEstimator(self.feature_columns, self.flags);


    def createDatasets(self):
        if self.mode == 'train':
            if not self.dataset_options_eval is None:
                dataset_maker_train = NeuralNetDatasetMaker('train', self.flags.model_dir, self.dataset_options_train);
                dataset_maker_eval = NeuralNetDatasetMaker('eval', self.flags.model_dir, self.dataset_options_eval);
                dataset_maker_train.createDatasetsAutoEncoder();
                dataset_maker_eval.createDatasetsAutoEncoder();
            else:
                print('model_dir: ' + str(self.flags.model_dir))
                dataset_maker = NeuralNetDatasetMaker('traineval', self.flags.model_dir, self.dataset_options_train);
                dataset_maker.createDatasetsAutoEncoder();
        elif self.mode == 'test':
            dataset_maker = NeuralNetDatasetMaker('test', self.flags.model_dir, self.dataset_options_test);
            dataset_maker.createDatasetsAutoEncoder();


    def train(self):
        self.createDatasets();

        if self.model is None:
            self._getModelEstimator();

        estimator = self.model.getEstimator();

        run_params = {
            'batch_size': self.flags.batch_size,
            'train_epochs': self.flags.train_epochs,
            'model_type': 'deep',
        }

        benchmark_logger = logger.config_benchmark_logger(self.flags)
        benchmark_logger.log_run_info('deep', 'Readmission Patient', run_params)

        # Train and evaluate the model every `flags.epochs_between_evals` epochs.
        for n in range(self.flags.train_epochs // self.flags.epochs_between_evals):
            estimator.train(input_fn=self._input_fn_train);
            # Display evaluation metrics
            tf.logging.info('Results at epoch %d / %d', (n + 1) * self.flags.epochs_between_evals, self.flags.train_epochs)
            tf.logging.info('-' * 60)

            results = estimator.predict(input_fn=self._input_fn_analyze);
            encodings = [p['encoding'] for p in results];
            basic_encodings = np.array(encodings);
            filename_basic_encodings = self.flags.model_dir + '/basic_encodings_' + str(n).zfill(5) + '.npy'
            np.save(filename_basic_encodings, basic_encodings);


    def analyze(self):
        if self.model is None:
            self._getModelEstimator();
        estimator = self.model.getEstimator();

        results = estimator.predict(input_fn=self._input_fn_analyze);
        encodings = [p['encoding'] for p in results];
        encodings = np.array(encodings);
        return encodings;


    def encode(self):
        self.createDatasets();

        if self.model is None:
            self._getModelEstimator();
        estimator = self.model.getEstimator();

        results_diag = estimator.predict(input_fn=self._input_fn_encode_diag);
        encodings_diag = [p['encoding'] for p in results_diag];
        encodings_diag = np.array(encodings_diag);

        results_main_diag = estimator.predict(input_fn=self._input_fn_encode_maindiag);
        encodings_main_diag = [p['encoding'] for p in results_main_diag];
        encodings_main_diag = np.array(encodings_main_diag);
        return [encodings_main_diag, encodings_diag];


    def getModelDir(self):
        return self.flags.model_dir;


    def getFlags(self):
        return self.flags;


    def getFilenameDatasetBalanced(self):
        if self.mode == 'train':
            return self.dataset_handler_train._getFilenameDatasetBalanced();
        elif self.mode == 'eval':
            return self.dataset_handler_eval._getFilenameDatasetBalanced();
        elif self.mode == 'test':
            return self.dataset_handler_test._getFilenameDatasetBalanced();
        else:
            print('unknown mode...exit')
            sys.exit();


    def getWeightsEmbeddingLayer(self, name_embedding):
        if name_embedding == 'main_diag':
            name_embedding_variable = self.feature_columns.getEmbeddingLayerNames()[0];
        elif name_embedding == 'diag':
            name_embedding_variable = self.feature_columns.getEmbeddingLayerNames()[1];
        else:
            print('embedding is unknown...exit')
            sys.exit();
        estimator = self.model.getEstimator();
        values = estimator.get_variable_value(name_embedding_variable)
        return values;



