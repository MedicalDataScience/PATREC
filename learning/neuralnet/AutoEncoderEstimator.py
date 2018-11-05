
import sys
import shutil

import tensorflow as tf
from tensorflow.python.summary import summary


class AutoEncoderEstimator:

    def __init__(self, feature_columns, flags):
        self.feature_columns = feature_columns;
        self.flags = flags;
        self.estimator = None;
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


    def _encoder(self, inputs, hidden_units, mode, dropout, batchnorm, feature_columns):
        print('hidden units: ' + str(hidden_units))
        input_layer = tf.feature_column.input_layer(inputs, feature_columns);
        for l_id, num_units in enumerate(hidden_units):
            l_name = 'encoder_hiddenlayer_%d' % l_id
            l = self._dense_batch_relu(input_layer, num_units, mode == tf.estimator.ModeKeys.TRAIN, l_name, batchnorm, dropout);
            self._add_hidden_layer_summary(l, l_name)
            input_layer = l;
        output = input_layer;
        return output;


    def _decoder(self, inputs, reverse_hidden_units, mode, dropout, batchnorm, feature_columns):
        print('reversed hidden units: ' + str(reverse_hidden_units))
        input_layer = inputs;
        for l_id, num_units in enumerate(reverse_hidden_units):
            l_name = 'decoder_hiddenlayer_%d' % l_id
            l = self._dense_batch_relu(input_layer, num_units, mode == tf.estimator.ModeKeys.TRAIN, l_name, batchnorm, dropout);
            self._add_hidden_layer_summary(l, l_name)
            input_layer = l;
        output = input_layer;
        return output;


    def _dense_batchnorm_fn(self, features, labels, mode, params):
        """Model function for Estimator."""
        hidden_units = params['hidden_units'];
        dropout = params['dropout']
        batchnorm = params['batchnorm']
        feature_columns = params['feature_columns'];
        reverse_hidden_units = hidden_units.copy();
        reverse_hidden_units.reverse();

        encoded = self._encoder(features, hidden_units, mode, dropout, batchnorm, feature_columns);
        logits = self._decoder(encoded, reverse_hidden_units, mode, dropout, batchnorm, feature_columns);

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
        loss = tf.losses.mean_squared_error(labels=labels, predictions=probabilities);

        if mode == tf.estimator.ModeKeys.EVAL:
            avg_loss = tf.reduce_mean(loss)
            tf.summary.scalar('avg_loss', avg_loss)

        eval_metric_ops = None;

        # learning_rate = params['learning_rate'];
        global_step = tf.train.get_global_step();
        starter_learning_rate = params['learning_rate'];
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000000, 0.96, staircase=True)
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


    def _build_estimator(self):
        tf.reset_default_graph()
        """Build an estimator appropriate for the given model type."""

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

        self.estimator = tf.estimator.Estimator(
            model_fn=self._dense_batchnorm_fn,
            model_dir=self.flags.model_dir,
            params=params_batchnorm,
            config=run_config,
            warm_start_from=ws
        )


    def getEstimator(self):
        if self.estimator is None:
            self._build_estimator();
        return self.estimator;