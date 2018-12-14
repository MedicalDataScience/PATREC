import sys
import shutil

import tensorflow as tf
from tensorflow.python.summary import summary

class NeuralNetEstimator:

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
