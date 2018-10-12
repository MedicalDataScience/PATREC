


import tensorflow as tf
from tensorflow.python.summary import summary


from utils.Dataset import Dataset


class NeuralNetModel:

    def __init__(self, dataset_options, feature_columns, flags, hidden_units):
        self.dataset_options = dataset_options;
        self.dataset = Dataset(self.dataset_options);
        self.feature_columns = feature_columns;
        self.flags = flags;
        self.hidden_units = hidden_units;
        self.model = None;
        self.num_samples_train = None;
        self.num_samples_validation = None;
        return;

    def _add_hidden_layer_summary(self, value, tag):
        summary.scalar('%s/fraction_of_zero_values' % tag, tf.nn.zero_fraction(value))
        summary.histogram('%s/activation' % tag, value)

    def _dense_batch_relu(self, input, num_nodes, phase, layer_name, batchnorm, dropout):
        if batchnorm:
            out = tf.layers.dense(input, num_nodes, activation=None, name=layer_name)
            out = tf.layers.batch_normalization(out, training=phase)
            out = tf.nn.relu(out);
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
        predicted = tf.argmax(tf.nn.sigmoid(logits), axis=1)
        predictions = tf.round(predicted);
        probabilities = tf.nn.sigmoid(logits);
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
        accuracy = tf.metrics.accuracy(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predictions, tf.float64),
                                       name='accuracy')
        precision = tf.metrics.precision(labels=tf.cast(labels, tf.float64),
                                         predictions=tf.cast(predictions, tf.float64), name='precision')
        recall = tf.metrics.recall(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predictions, tf.float64),
                                   name='recall')
        auc = tf.metrics.auc(labels=tf.cast(labels, tf.float64), predictions=probabilities[:, 1], name='auc')
        fp = tf.metrics.false_positives(labels=tf.cast(labels, tf.float64),
                                        predictions=tf.cast(predictions, tf.float64), name='false_positives')
        tp = tf.metrics.true_positives(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predictions, tf.float64),
                                       name='true_positives')
        fn = tf.metrics.false_negatives(labels=tf.cast(labels, tf.float64),
                                        predictions=tf.cast(predictions, tf.float64), name='false_negatives')
        tn = tf.metrics.true_negatives(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predictions, tf.float64),
                                       name='false_negatives')

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

        optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate']);
        # optimizer = tf.train.ProximalAdagradOptimizer(
        #    learning_rate=params['learning_rate'],
        #    l1_regularization_strength=0.001
        # )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            training_hooks=None,
            evaluation_hooks=None, )


    def _parse_csv(self, value):
        # print('Parsing', data_file)
        column_names = self.dataset.getColumnsData();
        default_values = self.feature_columns.getDefaultValues(column_names)
        columns = tf.decode_csv(value, record_defaults=default_values)
        features = dict(zip(column_names, columns))
        early_readmission_flagname = self.dataset_options.getEarlyReadmissionFlagname();
        labels = features.pop(early_readmission_flagname)
        # print('len(sorted(features.keys())): ' + str(len(sorted(features.keys()))))
        return features, tf.equal(labels, 1)


    def build_estimator(self):
        """Build an estimator appropriate for the given model type."""
        deep_columns = self.feature_columns.buildModelColumns()
        print('len(deep_columns): ' + str(len(deep_columns)))

        # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
        # trains faster than GPU for this model.
        run_config = tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(device_count={'GPU': 0}))

        print('batchnorm: ' + str(self.flags.batchnorm))
        print('dropout: ' + str(self.flags.dropout))
        print('learning rate: ' + str(self.flags.learningrate))
        print('batch size: ' + str(self.flags.batch_size))

        params_batchnorm = {'feature_columns': deep_columns,
                            'hidden_units': self.hidden_units,
                            'batchnorm': self.flags.batchnorm,
                            'dropout': self.flags.dropout,
                            'learning_rate': self.flags.learningrate};

        self.model = tf.estimator.Estimator(
            model_fn=self._dense_batchnorm_fn,
            model_dir=self.flags.model_dir,
            params=params_batchnorm,
            config=run_config,
        )
        return self.model;

    def input_fn(self, data_file, num_epochs, shuffle, batch_size):
        """Generate an input function for the Estimator."""
        assert tf.gfile.Exists(data_file), (
                '%s not found. Please make sure you have run data_download.py and '
                'set the --data_dir argument to the correct path.' % data_file)

        # Extract lines from input files using the Dataset API.
        # ds = tf.data.TextLineDataset(data_file)
        #
        # ds_pos = ds_pos.skip(1)
        # ds_neg = ds_neg.skip(1)
        # ds_neg = ds_neg.map(self._parse_csv, num_parallel_calls=5)
        # ds_pos = ds_pos.map(self._parse_csv, num_parallel_calls=5)
        #
        # dataset = tf.data.Dataset.zip((ds_pos, ds_neg))
        #
        # # Each input element will be converted into a two-element `Dataset` using
        # # `Dataset.from_tensors()` and `Dataset.concatenate()`, then `Dataset.flat_map()`
        # # will flatten the resulting `Dataset`s into a single `Dataset`.
        # dataset = dataset.flat_map(
        #     lambda ex_pos, ex_neg: tf.data.Dataset.from_tensors(ex_pos).concatenate(
        #         tf.data.Dataset.from_tensors(ex_neg)))
        #
        # if shuffle:
        #     dataset = dataset.shuffle(buffer_size=self.num_samples_train)
        #
        # # We call repeat after shuffling, rather than before, to prevent separate
        # # epochs from blending together.
        # dataset = dataset.repeat(num_epochs)
        # dataset = dataset.batch(batch_size)

        dataset = tf.data.TextLineDataset(data_file)
        dataset = dataset.skip(1)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.num_samples_train)

        dataset = dataset.map(self._parse_csv, num_parallel_calls=5)

        # We call repeat after shuffling, rather than before, to prevent separate
        # epochs from blending together.
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        return dataset


    def getFilenameDatasets(self):
        [df_training, df_testing] = self.dataset.getBalancedSubsetTrainingAndTesting();
        self.num_samples_train = df_training.shape[0];
        self.num_samples_validation = df_testing.shape[0];
        filename_dataset_base = self.dataset_options.getFilename();
        filename_training = filename_dataset_base[:-4] + '_balanced_training.csv'
        filename_testing = filename_dataset_base[:-4] + '_balanced_testing.csv'
        df_training.to_csv(filename_training, line_terminator='\n', index=False);
        df_testing.to_csv(filename_testing, line_terminator='\n', index=False);
        return [filename_training, filename_testing]


    def setModelDir(self):
        modeldir_base = self.flags.model_dir;
        dropoutrate = self.flags.dropout;
        learningrate = self.flags.learningrate;
        batchnorm = self.flags.batchnorm;
        batchsize = self.flags.batch_size;
        data_prefix = self.dataset_options.getDataPrefix();

        suffix_modeldir = data_prefix + '_' + str(self.hidden_units[0]);
        for k in range(1, len(self.hidden_units)):
            suffix_modeldir = suffix_modeldir + '_' + str(self.hidden_units[k]);

        suffix_modeldir = suffix_modeldir + '_dropout_' + str(dropoutrate);
        suffix_modeldir = suffix_modeldir + '_learningrate_' + str(learningrate);
        suffix_modeldir = suffix_modeldir + '_batchnorm_' + str(batchnorm);
        suffix_modeldir = suffix_modeldir + '_batchsize_' + str(batchsize);
        model_dir = modeldir_base + '/' + suffix_modeldir;
        self.flags.model_dir = model_dir;


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
        print('r: ' + str(r))
