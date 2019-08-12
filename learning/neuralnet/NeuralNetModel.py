import os
import sys
import shutil
import numpy as np

import tensorflow as tf
from tensorflow.python.summary import summary

from learning.neuralnet.NeuralNetEstimator import NeuralNetEstimator
from learning.neuralnet.NeuralNetDatasetHandler import NeuralNetDatasetHandler
from learning.neuralnet.NeuralNetDatasetMaker import NeuralNetDatasetMaker
from utils.Dataset import Dataset

from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers


class NeuralNetModel():

    def __init__(self, mode, dict_dataset_options, feature_columns, flags):
        self.feature_columns = feature_columns;
        self.dataset_options_train = dict_dataset_options['train'];
        self.dataset_options_eval = dict_dataset_options['eval'];
        self.dataset_options_test = dict_dataset_options['test'];
        self.mode = mode;
        self.flags = flags;

        if not os.path.exists(self.flags.model_dir):
            os.makedirs(self.flags.model_dir)

        if self.mode == 'train':
            if self.dataset_options_eval is not None:
                self.dataset_handler_train = NeuralNetDatasetHandler(self.flags.model_dir, self.dataset_options_train,
                                                                     feature_columns, 'train',
                                                                     self.dataset_options_train);
                self.dataset_handler_eval = NeuralNetDatasetHandler(self.flags.model_dir, self.dataset_options_eval,
                                                                    feature_columns, 'eval');
            else:
                self.dataset_handler_train = NeuralNetDatasetHandler(self.flags.model_dir, self.dataset_options_train,
                                                                     feature_columns, 'train');
                self.dataset_handler_eval = NeuralNetDatasetHandler(self.flags.model_dir, self.dataset_options_train,
                                                                    feature_columns, 'eval');
        elif self.mode == 'test':
            self.dataset_handler_test = NeuralNetDatasetHandler(self.dataset_options_test, feature_columns, 'test');

        self.model = None;
        self.flags.hidden_units = [int(u) for u in self.flags.hidden_units];
        return;

    def _setModelDir(self):
        modeldir_base = self.flags.model_dir;
        dropoutrate = self.flags.dropout;
        learningrate = self.flags.learningrate;
        batchnorm = self.flags.batchnorm;
        batchsize = self.flags.batch_size;
        dataset_filename_options = self.dataset_options_train.getFilenameOptions(filteroptions=False);
        suffix_modeldir = '';
        if self.flags.continue_training:
            suffix_modeldir = 'warmstart_';
        suffix_modeldir = suffix_modeldir + dataset_filename_options + '_' + str(self.flags.hidden_units[0]);

        for k in range(1, len(self.flags.hidden_units)):
            suffix_modeldir = suffix_modeldir + '_' + str(self.flags.hidden_units[k]);

        suffix_modeldir = suffix_modeldir + '_dropout_' + str(dropoutrate);
        suffix_modeldir = suffix_modeldir + '_learningrate_' + str(learningrate);
        suffix_modeldir = suffix_modeldir + '_batchnorm_' + str(batchnorm);
        suffix_modeldir = suffix_modeldir + '_batchsize_' + str(batchsize);

        # Add filtering option if specified
        if self.dataset_handler_train.dataset.options.options_filtering is not None:
            suffix_modeldir += '_filtering_' + str(self.dataset_handler_train.dataset.options.options_filtering)

        model_dir = modeldir_base + '/' + suffix_modeldir;
        self.flags.model_dir = model_dir;

    def _input_fn_train(self):
        dataset = self.dataset_handler_train.readDatasetTF();
        dataset = dataset.repeat(self.flags.epochs_between_evals);
        dataset = dataset.batch(self.flags.batch_size);
        return dataset;

    def _input_fn_eval(self):
        dataset = self.dataset_handler_eval.readDatasetTF();
        dataset = dataset.repeat(1);
        dataset = dataset.batch(self.flags.batch_size);
        return dataset;

    def _input_fn_test(self):
        dataset = self.dataset_handler_test.readDatasetTF();
        dataset = dataset.repeat(1);
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
            if self.mode == 'train':
                self._setModelDir();
                # Clean up the model directory if present
                # if not self.flags.model_dir == self.flags.pretrained_model_dir:
                #     shutil.rmtree(self.flags.model_dir, ignore_errors=True)
            self.model = NeuralNetEstimator(self.feature_columns, self.flags,
                                            self.dataset_handler_train.dataset.getNumSamplesBalancedSubset());

    def createDatasets(self):
        if self.mode == 'train':
            if self.dataset_options_eval is not None:
                dataset_maker_train = NeuralNetDatasetMaker('train', self.flags.model_dir, self.dataset_options_train);
                dataset_maker_eval = NeuralNetDatasetMaker('eval', self.flags.model_dir, self.dataset_options_eval);
                dataset_maker_train.createDatasets();
                dataset_maker_eval.createDatasets();
            else:
                # dataset_maker = NeuralNetDatasetMaker('traineval', self.flags.model_dir, self.dataset_options_train, balanced_datasets=False);
                dataset_maker = NeuralNetDatasetMaker('traineval', self.flags.model_dir, self.dataset_options_train);
                dataset_maker.createDatasets();
        elif self.mode == 'test':
            dataset_maker = NeuralNetDatasetMaker('test', self.flags.model_dir, self.dataset_options_test);
            dataset_maker.createDatasets();

    def train(self):

        self.createDatasets();

        if self.model is None:
            self._getModelEstimator();

        estimator = self.model.getEstimator();

        # def train_input_fn():
        #     return self.input_fn(self.flags.epochs_between_evals, True, self.flags.batch_size, 'train')
        #
        # def eval_input_fn():
        #     return self.input_fn(1, False, self.flags.batch_size, 'eval')

        run_params = {
            'batch_size': self.flags.batch_size,
            'train_epochs': self.flags.train_epochs,
            'model_type': 'deep',
        }

        benchmark_logger = logger.config_benchmark_logger(self.flags)
        benchmark_logger.log_run_info('deep', 'Readmission Patient', run_params)

        # Train and evaluate the model every `flags.epochs_between_evals` epochs.
        for n in range(self.flags.train_epochs // self.flags.epochs_between_evals):
            print('n: ' + str(n))
            estimator.train(input_fn=self._input_fn_train)
            results = estimator.evaluate(input_fn=self._input_fn_eval)
            # Display evaluation metrics
            tf.logging.info('Results at epoch %d / %d', (n + 1) * self.flags.epochs_between_evals,
                            self.flags.train_epochs)
            tf.logging.info('-' * 60)

            for key in sorted(results):
                tf.logging.info('%s: %s' % (key, results[key]))

            benchmark_logger.log_evaluation_result(results)

            if model_helpers.past_stop_threshold(self.flags.stop_threshold, results['accuracy']):
                break

            # Export the model
            print('export the model?')
            if n % 10 == 0 and self.flags.export_dir is not None:
                self.export_model()

            # Break training loop if DP is enabled and the privacy budget has been used up
            if estimator.privacy_budget_exceeded():
                print("Privacy budget met, stopping training")
                break

    def predict(self):
        if self.model is None:
            self._getModelEstimator();
        estimator = self.model.getEstimator();
        results = estimator.predict(input_fn=self._input_fn_test)
        return results;

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
