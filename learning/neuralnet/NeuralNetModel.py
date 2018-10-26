import os
import sys
import shutil
import numpy as np

import tensorflow as tf
from tensorflow.python.summary import summary


from utils.Dataset import Dataset


class NeuralNetModel():

    def __init__(self, dataset_options, feature_columns, flags, balanced_datasets=True):
        self.dataset_options = dataset_options;
        self.dataset = Dataset(self.dataset_options);
        self.feature_columns = feature_columns;
        self.flags = flags;
        self.model = None;
        self.num_samples_train = None;
        self.num_samples_validation = None;
        self.balanced_datasets = balanced_datasets;

        self.flags.hidden_units = [int(u) for u in self.flags.hidden_units];
        return;


    def _add_hidden_layer_summary(self, value, tag):
        summary.scalar('%s/fraction_of_zero_values' % tag, tf.nn.zero_fraction(value))
        summary.histogram('%s/activation' % tag, value)


    def _dense_batch_relu(self, input, num_nodes, phase, layer_name, batchnorm, dropout):
        if batchnorm:
            out = tf.layers.dense(input, num_nodes, activation=tf.nn.relu, name=layer_name)
            out = tf.layers.batch_normalization(out, training=phase)
        else:
            out = tf.layers.dense(input, num_nodes, activation=tf.nn.relu, name=layer_name)

        if dropout is not None:
            out = tf.layers.dropout(out, rate=dropout, training=phase)
        return out;


    def _dense_batchnorm_fn(self, features, labels, mode, params):
        """Model function for Estimator."""
        hidden_units = params['hidden_units'];
        dropout = params['dropout']
        batchnorm = params['batchnorm']

        input_layer = tf.feature_column.input_layer(features, params['feature_columns']);
        for l_id, num_units in enumerate(hidden_units):
            l_name = 'hiddenlayer_%d' % l_id
            l = self._dense_batch_relu(input_layer, num_units, mode == tf.estimator.ModeKeys.TRAIN, l_name, batchnorm,
                                 dropout);
            self._add_hidden_layer_summary(l, l_name)
            input_layer = l;

        if batchnorm:
            logits = tf.layers.dense(input_layer, 2, activation=None, name='logits')
            logits = tf.layers.batch_normalization(logits, training=(mode == tf.estimator.ModeKeys.TRAIN));
        else:
            logits = tf.layers.dense(input_layer, 2, activation=None, name='logits')
        self._add_hidden_layer_summary(logits, 'logits')

        # logits = tf.reshape(logits, [-1]);
        # labels = tf.reshape(labels, [-1]);

        # Reshape output layer to 1-dim Tensor to return predictions
        probabilities = tf.nn.softmax(logits);
        predictions = tf.round(probabilities);
        predicted = tf.argmax(predictions, axis=1)
        # Provide an estimator spec for `ModeKeys.PREDICT`.
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                export_outputs={'predict_output': tf.estimator.export.PredictOutput({"Wiederkehrer": predictions,
                                                                                     'probabilities': probabilities})},
                predictions={
                    'Wiederkehrer': predictions,
                    'logits': logits,
                    'probabilities': probabilities
                })

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32), logits=logits);
        loss = tf.reduce_mean(cross_entropy)

        # Compute evaluation metrics.
        # mean_squared_error = tf.metrics.mean_squared_error(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predictions, tf.float64), name='mean_squared_error')
        accuracy = tf.metrics.accuracy(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predicted, tf.float64), name='accuracy')
        precision = tf.metrics.precision(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predicted, tf.float64), name='precision')
        recall = tf.metrics.recall(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predicted, tf.float64), name='recall')
        auc = tf.metrics.auc(labels=tf.cast(labels, tf.float64), predictions=probabilities[:, 1], name='auc')
        fp = tf.metrics.false_positives(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predicted, tf.float64), name='false_positives')
        tp = tf.metrics.true_positives(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predicted, tf.float64), name='true_positives')
        fn = tf.metrics.false_negatives(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predicted, tf.float64), name='false_negatives')
        tn = tf.metrics.true_negatives(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predicted, tf.float64), name='false_negatives')

        tf.summary.scalar('accuracy', accuracy[1]);

        if mode == tf.estimator.ModeKeys.EVAL:
            avg_loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('avg_loss', avg_loss)

        # Calculate root mean squared error as additional eval metric
        eval_metric_ops = {'accuracy': accuracy,
                           'precision': precision,
                           'recall': recall,
                           'auc': auc,
                           'true positives': tp,
                           'true negatives': tn,
                           'false positives': fp,
                           'false negatives': fn
                           }

        # learning_rate = params['learning_rate'];
        global_step = tf.train.get_global_step();
        starter_learning_rate = params['learning_rate'];
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate);
        # optimizer = tf.train.ProximalAdagradOptimizer(
        #    learning_rate=params['learning_rate'],
        #    l1_regularization_strength=0.001
        # )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=global_step)

        # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            training_hooks=None,
            evaluation_hooks=None, )


    def _setModelDir(self):
        modeldir_base = self.flags.model_dir;
        dropoutrate = self.flags.dropout;
        learningrate = self.flags.learningrate;
        batchnorm = self.flags.batchnorm;
        batchsize = self.flags.batch_size;
        dataset_filename_options = self.dataset_options.getFilenameOptions(filteroptions=False);
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
        model_dir = modeldir_base + '/' + suffix_modeldir;
        self.flags.model_dir = model_dir;


    def _parse_csv(self, value):
        # print('Parsing', data_file)
        column_names = self.dataset.getColumnsData();
        default_values = self.feature_columns.getDefaultValues(column_names)
        columns = tf.decode_csv(value, record_defaults=default_values)
        features = dict(zip(column_names, columns))
        early_readmission_flagname = self.dataset_options.getEarlyReadmissionFlagname();
        labels = features.pop(early_readmission_flagname)
        return features, tf.equal(labels, 1)


    def _build_estimator(self, mode):
        tf.reset_default_graph()
        """Build an estimator appropriate for the given model type."""
        if mode != 'test' and mode != 'analyze':
            self._setModelDir();
            # Clean up the model directory if present
            if not self.flags.model_dir == self.flags.pretrained_model_dir:
                shutil.rmtree(self.flags.model_dir, ignore_errors=True)
        print(self.flags.model_dir)
        print(self.flags.pretrained_model_dir)
        deep_columns = self.feature_columns.buildModelColumns()

        # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
        # trains faster than GPU for this model.
        run_config = tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(device_count={'GPU': 0}))

        #warm start settings:
        ws = None;
        if self.flags.continue_training:
            # like that: all weights (input layer and hidden weights are warmstarted)
            if self.flags.pretrained_model_dir is not None:
                # os.system('scp -r ' + self.flags.pretrained_model_dir + ' ' + self.flags.model_dir + '/')
                ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=self.flags.pretrained_model_dir,
                                                    var_name_to_prev_var_name=self.feature_columns.getConversionDict()
                                                    )
            else:
                print('continue_training flag is set to True, but not pretrained_model_dir_specified...exit')
                sys.exit();


        params_batchnorm = {'feature_columns': deep_columns,
                            'hidden_units': self.flags.hidden_units,
                            'batchnorm': self.flags.batchnorm,
                            'dropout': self.flags.dropout,
                            'learning_rate': self.flags.learningrate};

        self.model = tf.estimator.Estimator(
            model_fn=self._dense_batchnorm_fn,
            model_dir=self.flags.model_dir,
            params=params_batchnorm,
            config=run_config,
            warm_start_from=ws
        )
        return self.model;


    def _getFilenameDatasetBalanced(self, mode):
        filename_dataset_base = self.dataset_options.getFilename();
        if mode == 'train':
            filename_train = filename_dataset_base[:-4] + '_balanced_train.csv'
            filename = filename_train;
        elif mode == 'eval':
            filename_eval = filename_dataset_base[:-4] + '_balanced_eval.csv'
            filename = filename_eval;
        elif mode == 'test':
            filename_test = filename_dataset_base[:-4] + '_balanced_test.csv'
            filename = filename_test;
        else:
            print('unknown mode...exit')
            sys.exit();
        return filename;


    def _getFilenamesDatasetAll(self, mode):
        filename_dataset_base = self.dataset_options.getFilename();
        if mode == 'train':
            filename_train_pos = filename_dataset_base[:-4] + '_train_pos.csv'
            filename_train_neg = filename_dataset_base[:-4] + '_train_neg.csv'
            filenames = [filename_train_pos, filename_train_neg];
        elif mode == 'eval':
            filename_eval_pos = filename_dataset_base[:-4] + '_eval_pos.csv'
            filename_eval_neg = filename_dataset_base[:-4] + '_eval_neg.csv'
            filenames = [filename_eval_pos, filename_eval_neg];
        elif mode == 'test':
            filename_test_pos = filename_dataset_base[:-4] + '_test_pos.csv'
            filename_test_neg = filename_dataset_base[:-4] + '_test_neg.csv'
            filenames = [filename_test_pos, filename_test_neg];
        else:
            print('unknown mode...exit')
            sys.exit();
        return filenames;


    def _dataset_reader(self, shuffle, mode):
        if self.balanced_datasets:
            filename_dataset = self._getFilenameDatasetBalanced(mode);
            # shuffle is only performed for training; not optimal --> maybe five another flag to specify training/eval
            dataset = tf.data.TextLineDataset(filename_dataset)
            dataset = dataset.skip(1)
            if shuffle:
                dataset = dataset.shuffle(buffer_size=self.num_samples_train)

            dataset = dataset.map(self._parse_csv, num_parallel_calls=5)
            return dataset;
        else:
            filenames_dataset = self._getFilenamesDatasetAll(mode);
            if shuffle:
                data_file_pos = filenames_dataset[0];
                data_file_neg = filenames_dataset[1];
            else:
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
            if shuffle:
                dataset = dataset.shuffle(buffer_size=self.num_samples_train)
            return dataset;


    def input_fn(self, num_epochs, shuffle, batch_size, mode):
        """Generate an input function for the Estimator."""
        print('input fn: ' + str(mode))
        dataset = self._dataset_reader(shuffle, mode);
        # We call repeat after shuffling, rather than before, to prevent separate
        # epochs from blending together.
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        return dataset


    def export_model(self):
        """Export to SavedModel format.
        Args:
        model: Estimator object
        export_dir: directory to export the model.
        """
        deep_columns = self.feature_columns.buildModelColumns()
        feature_spec = tf.feature_column.make_parse_example_spec(deep_columns)
        example_input_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
        r = self.model.export_savedmodel(self.flags.export_dir, example_input_fn)


    def getModel(self, mode=None):
        if self.model is None:
            self._build_estimator(mode);
        return self.model;


    def getModelDir(self):
        return self.flags.model_dir;

    def getFlags(self):
        return self.flags;


    def createDatasets(self, mode=None):
        print('_getFilenameDatasetBalanced: ' + str(mode))
        filename_dataset_base = self.dataset_options.getFilename();

        if mode == 'test':
            print('get balanced dataset for testing...')
            df_balanced = self.dataset.getBalancedSubSet();
            filename_test = filename_dataset_base[:-4] + '_balanced_test.csv'
            df_balanced.to_csv(filename_test, line_terminator='\n', index=False);
            print(filename_test)
        else:
            if self.balanced_datasets:
                [df_training, df_testing] = self.dataset.getBalancedSubsetTrainingAndTesting();
                self.num_samples_train = df_training.shape[0];
                self.num_samples_validation = df_testing.shape[0];
                filename_train = filename_dataset_base[:-4] + '_balanced_train.csv'
                filename_eval = filename_dataset_base[:-4] + '_balanced_eval.csv'
                df_training.to_csv(filename_train, line_terminator='\n', index=False);
                df_testing.to_csv(filename_eval, line_terminator='\n', index=False);
                print(filename_train)
                print(filename_eval)
            else:
                [training, testing] = self.dataset.getTrainingAndTestingSet();
                df_training_pos = training[0];
                df_training_neg = training[1];
                df_eval_pos = testing[0];
                df_eval_neg = testing[1];
                self.num_samples_train = 2*int(df_training_neg.shape[0]);
                self.num_samples_validation = 2*int(df_eval_neg.shape[0]);
                filename_train_pos = filename_dataset_base[:-4] + '_train_pos.csv'
                filename_train_neg = filename_dataset_base[:-4] + '_train_neg.csv'
                filename_eval_pos = filename_dataset_base[:-4] + '_eval_pos.csv'
                filename_eval_neg = filename_dataset_base[:-4] + '_eval_neg.csv'
                df_training_pos.to_csv(filename_train_pos, line_terminator='\n', index=False);
                df_training_neg.to_csv(filename_train_neg, line_terminator='\n', index=False);
                df_eval_pos.to_csv(filename_eval_pos, line_terminator='\n', index=False);
                df_eval_neg.to_csv(filename_eval_neg, line_terminator='\n', index=False);


    def getFilenameDatasetBalanced(self, mode):
        return self._getFilenameDatasetBalanced(mode);

    def getFilenamesDatasetAll(self, mode):
        return self._getFilenamesDatasetAll(mode)


    def getWeightsEmbeddingLayer(self, name_embedding):
        if name_embedding == 'main_diag':
            name_embedding_variable = self.feature_columns.getEmbeddingLayerNames()[0];
        elif name_embedding == 'diag':
            name_embedding_variable = self.feature_columns.getEmbeddingLayerNames()[1];
        else:
            print('embedding is unknown...exit')
            sys.exit();
        #var = [v for v in self.model.get_variable_names() if v.name == name_embedding_variable][0]
        values = self.model.get_variable_value(name_embedding_variable)
        return values;



