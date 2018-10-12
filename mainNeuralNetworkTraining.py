# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

tf_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models';

print(tf_base_dir)
if not tf_base_dir in sys.path:
    sys.path.append(tf_base_dir);

DIRPROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';

from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers


from learning.neuralnet.NeuralNetModel import NeuralNetModel
from learning.neuralnet.FeatureColumnsPatrec import FeatureColumnsPatrec
from learning.neuralnet.FeatureColumnsNZ import FeatureColumnsNZ
from utils.DatasetOptions import DatasetOptions
from utils.DatasetOptionsNZ_OLD import DatasetOptionsNZ
from utils.Dataset import Dataset


# hidden_units = [100, 100, 50, 50, 25, 25, 25, 25, 10, 10, 10];
# hidden_units = [35, 35, 35, 35, 35, 35, 35, 35, 35, 35];
# hidden_units = [120, 100, 80, 60, 40, 10]
hidden_units = [100, 80, 40, 40, 40, 10];
# dropout_rate = 0.15;
# batch_norm = True;
# learning_rate = 0.05;



def define_flags():
    """Add supervised learning flags, as well as wide-deep model type."""
    flags_core.define_base()
    flags_core.define_benchmark()

    flags.adopt_module_key_flags(flags_core)

    flags_core.set_defaults(data_dir=DIRPROJECT + 'data/',
                            model_dir='/tmp/patients_model',
                            export_dir='/tmp/patients_model/export_model',
                            train_epochs=250,
                            epochs_between_evals=1,
                            batch_size=160)



def run_deep(flags_obj):
    """Run Wide-Deep training and eval loop.
    Args:
    flags_obj: An object containing parsed flag values.
    """
    dict_data = {
        'dir_data':         DIRPROJECT + 'data/',
        'data_prefix':      'nz',
        'dataset':          '20122016',
        'encoding':         'embedding',
        'featureset':       'standard'
    }
    dataset_options = DatasetOptions(dict_data);
    feature_columns_patrec = FeatureColumnsPatrec(dataset_options=dataset_options)
    feature_columns_nz = FeatureColumnsNZ(dataset_options=dataset_options);

    feature_columns = feature_columns_nz;
    nn = NeuralNetModel(dataset_options, feature_columns, flags_obj, hidden_units);
    nn.setModelDir()

    # Clean up the model directory if present
    if not flags_obj.continue_training:
        shutil.rmtree(flags_obj.model_dir, ignore_errors=True)

    filenames_dataset = nn.getFilenameDatasets();
    train_file = filenames_dataset[0];
    test_file = filenames_dataset[1];

    model = nn.build_estimator()
    # Train and evaluate the model every `flags.epochs_between_evals` epochs.
    def train_input_fn():
        return nn.input_fn(train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size)

    def eval_input_fn():
        return nn.input_fn(test_file, 1, False, flags_obj.batch_size)

    run_params = {
      'batch_size': flags_obj.batch_size,
      'train_epochs': flags_obj.train_epochs,
      'model_type': 'deep',
    }

    benchmark_logger = logger.config_benchmark_logger(flags_obj)
    benchmark_logger.log_run_info('deep', 'Readmission Patient', run_params)

    # Train and evaluate the model every `flags.epochs_between_evals` epochs.
    for n in range(flags_obj.train_epochs // flags_obj.epochs_between_evals):
        print('n: ' + str(n))
        model.train(input_fn=train_input_fn, steps=500)
        results = model.evaluate(input_fn=eval_input_fn)
        # Display evaluation metrics
        tf.logging.info('Results at epoch %d / %d',
                        (n + 1) * flags_obj.epochs_between_evals,
                        flags_obj.train_epochs)
        tf.logging.info('-' * 60)

        for key in sorted(results):
            tf.logging.info('%s: %s' % (key, results[key]))

        benchmark_logger.log_evaluation_result(results)

        if model_helpers.past_stop_threshold(flags_obj.stop_threshold, results['accuracy']):
            break;

        # Export the model
        print('export the model?')
        if n%10==0 and flags_obj.export_dir is not None:
            print('export model: ' + str(flags_obj.export_dir))
            nn.export_model()


def main(_):
    run_deep(flags.FLAGS)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    define_flags()
    absl_app.run(main)
