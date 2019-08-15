import sys
import shutil

import tensorflow as tf
from tensorflow.python.summary import summary
from privacy.analysis import privacy_ledger
from privacy.analysis.rdp_accountant import compute_rdp_from_ledger, get_privacy_spent
from privacy.optimizers import dp_optimizer


class CheckPrivacyBudgetHook(tf.estimator.SessionRunHook):
    def __init__(self, ledger, target_epsilon, target_delta, update_op, update_op_placeholder, privacy_exceeded_list):
        self._samples, self._queries = ledger.get_unformatted_ledger()
        self._target_epsilon = target_epsilon
        self._target_delta = target_delta
        self._update_op = update_op
        self._update_op_placeholder = update_op_placeholder
        self._executed = False
        self._privacy_exceeded_list = privacy_exceeded_list

    # def end(self, session):
    def after_run(self, run_context, run_values):
        if not self._executed:
            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
            samples = run_context.session.run(self._samples)
            queries = run_context.session.run(self._queries)
            formatted_ledger = privacy_ledger.format_ledger(samples, queries)
            rdp = compute_rdp_from_ledger(formatted_ledger, orders)
            eps = get_privacy_spent(orders, rdp, target_delta=self._target_delta)[0]
            print('For delta={:.5}, the current epsilon is: {:.5}'.format(self._target_delta, eps))
            run_context.session.run(self._update_op, feed_dict={self._update_op_placeholder: eps})
            self._executed = True

            if eps >= self._target_epsilon:
                print("Target epsilon met or exceeded: {:.5}".format(eps))
                # run_context.request_stop()

                # Inform the model that the privacy budget has been exceeded
                self._privacy_exceeded_list[0] = True


class NeuralNetEstimator:

    def __init__(self, feature_columns, flags, training_samples_count):
        self.feature_columns = feature_columns
        self.flags = flags
        self.estimator = None

        # DP necessary members
        self.training_samples_count = training_samples_count
        self.q = self.flags.batch_size * 1.0 / self.training_samples_count
        self.privacy_budget_exceeded = [False]
        return

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
        return out

    def _dense_batchnorm_fn(self, features, labels, mode, params):
        if self.flags.enable_dp:
            assert not self.is_privacy_budget_exceeded(), "Attempt to train model after privacy budget exceeded"

        """Model function for Estimator."""
        hidden_units = params['hidden_units']
        dropout = params['dropout']
        batchnorm = params['batchnorm']

        input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
        for l_id, num_units in enumerate(hidden_units):
            l_name = 'hiddenlayer_%d' % l_id
            l = self._dense_batch_relu(input_layer, num_units, mode == tf.estimator.ModeKeys.TRAIN, l_name, batchnorm,
                                       dropout)
            self._add_hidden_layer_summary(l, l_name)
            input_layer = l

        if batchnorm:
            logits = tf.layers.dense(input_layer, 2, activation=None, name='logits')
            logits = tf.layers.batch_normalization(logits, training=(mode == tf.estimator.ModeKeys.TRAIN))
        else:
            logits = tf.layers.dense(input_layer, 2, activation=None, name='logits')
        self._add_hidden_layer_summary(logits, 'logits')

        # Reshape output layer to 1-dim Tensor to return predictions
        probabilities = tf.nn.softmax(logits)
        predictions = tf.round(probabilities)
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

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32), logits=logits)
        scalar_loss = tf.reduce_mean(cross_entropy)

        if self.flags.enable_dp:
            training_loss = cross_entropy
        else:
            training_loss = scalar_loss

        # Compute evaluation metrics.
        # mean_squared_error = tf.metrics.mean_squared_error(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predictions, tf.float64), name='mean_squared_error')
        accuracy = tf.metrics.accuracy(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predicted, tf.float64),
                                       name='accuracy')
        precision = tf.metrics.precision(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predicted, tf.float64),
                                         name='precision')
        recall = tf.metrics.recall(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predicted, tf.float64),
                                   name='recall')
        auc = tf.metrics.auc(labels=tf.cast(labels, tf.float64), predictions=probabilities[:, 1], name='auc')
        fp = tf.metrics.false_positives(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predicted, tf.float64),
                                        name='false_positives')
        tp = tf.metrics.true_positives(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predicted, tf.float64),
                                       name='true_positives')
        fn = tf.metrics.false_negatives(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predicted, tf.float64),
                                        name='false_negatives')
        tn = tf.metrics.true_negatives(labels=tf.cast(labels, tf.float64), predictions=tf.cast(predicted, tf.float64),
                                       name='false_negatives')

        tf.summary.scalar('accuracy', accuracy[1])

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

        global_step = tf.train.get_global_step()
        starter_learning_rate = params['learning_rate']
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000000, 0.96, staircase=True)

        # If differential privacy is enabled, use it
        if self.flags.enable_dp:
            ledger = privacy_ledger.PrivacyLedger(population_size=self.training_samples_count,
                                                  selection_probability=self.q)
            optimizer = dp_optimizer.DPAdamGaussianOptimizer(learning_rate=learning_rate, l2_norm_clip=self.flags.dp_c,
                                                             noise_multiplier=self.flags.dp_sigma,
                                                             num_microbatches=self.flags.dp_num_microbatches,
                                                             ledger=ledger)

            # Add summary for DP-Epsilon
            dp_eps = tf.Variable(0, trainable=False, dtype=tf.float32)
            dp_eps_placeholder = tf.placeholder(tf.float32)
            update_dp_eps_op = tf.assign(dp_eps, dp_eps_placeholder)
            eval_metric_ops['DP-Epsilon'] = dp_eps, tf.metrics.mean(dp_eps)[1]

            eval_hooks = [CheckPrivacyBudgetHook(ledger, self.flags.dp_eps, self.flags.dp_delta, update_dp_eps_op,
                                                 dp_eps_placeholder, self.privacy_budget_exceeded)]

        # Otherwise just use a normal ADAMOptimizer
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            eval_hooks = None

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=training_loss, global_step=global_step)

        # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=scalar_loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            training_hooks=None,
            evaluation_hooks=eval_hooks)

    def _build_estimator(self):
        tf.reset_default_graph()
        """Build an estimator appropriate for the given model type."""

        deep_columns = self.feature_columns.buildModelColumns()

        # Specify whether to use CPU or GPU
        if self.flags.force_cpu:
            run_config = tf.estimator.RunConfig().replace(
                session_config=tf.ConfigProto(device_count={'GPU': 0}))
        else:
            run_config = tf.estimator.RunConfig()

        # warm start settings:
        ws = None
        if self.flags.continue_training:
            # like that: all weights (input layer and hidden weights are warmstarted)
            if self.flags.pretrained_model_dir is not None:
                # os.system('scp -r ' + self.flags.pretrained_model_dir + ' ' + self.flags.model_dir + '/')
                ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=self.flags.pretrained_model_dir,
                                                    var_name_to_prev_var_name=self.feature_columns.getConversionDict()
                                                    )
            else:
                print('continue_training flag is set to True, but not pretrained_model_dir_specified...exit')
                sys.exit()

        params_batchnorm = {'feature_columns': deep_columns,
                            'hidden_units': self.flags.hidden_units,
                            'batchnorm': self.flags.batchnorm,
                            'dropout': self.flags.dropout,
                            'learning_rate': self.flags.learningrate}

        self.estimator = tf.estimator.Estimator(
            model_fn=self._dense_batchnorm_fn,
            model_dir=self.flags.model_dir,
            params=params_batchnorm,
            config=run_config,
            warm_start_from=ws
        )

    def getEstimator(self):
        if self.estimator is None:
            self._build_estimator()
        return self.estimator

    def is_privacy_budget_exceeded(self):
        return self.privacy_budget_exceeded[0]
