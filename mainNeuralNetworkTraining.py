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
from learning.neuralnet.FeatureColumnsNZFusion import FeatureColumnsNZFusion
from learning.neuralnet.FeatureColumnsPatrecFusion import FeatureColumnsPatrecFusion
from utils.DatasetOptions import DatasetOptions


# hidden_units = [100, 100, 50, 50, 25, 25, 25, 25, 10, 10, 10];
# hidden_units = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30];
# hidden_units = [100, 80, 60, 40, 10]
#hidden_units = [20, 20, 20, 10, 10];
#hidden_units = [10, 10, 10, 10, 10];
# hidden_units = [60, 60, 40, 40, 20, 20, 20, 10]
# hidden_units = [60, 40, 20, 10, 10]

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
        'dir_data':             DIRPROJECT + 'data/',
        'data_prefix':          'patrec',
        'dataset':              '20122015',
        'encoding':             'embedding',
        'newfeatures':          None,
        'featurereduction':     {'method': 'FUSION'},
        'grouping':             'verylightgrouping'
    }
    dataset_options = DatasetOptions(dict_data);
    # feature_columns_patrec = FeatureColumnsPatrec(dataset_options=dataset_options)
    # feature_columns_nz = FeatureColumnsNZ(dataset_options=dataset_options);

    if dict_data['data_prefix'] == 'nz':
        feature_columns_nz_fusion = FeatureColumnsNZFusion(dataset_options=dataset_options);
        feature_columns = feature_columns_nz_fusion;
    elif dict_data['data_prefix'] == 'patrec':
        feature_columns_patrec_fusion = FeatureColumnsPatrecFusion(dataset_options=dataset_options);
        feature_columns = feature_columns_patrec_fusion;
    else:
        print('unknown data prefix..exit')
        sys.exit()

    nn = NeuralNetModel(dataset_options, feature_columns, flags_obj, balanced_datasets=True);
    nn.createDatasets();
    model = nn.getModel()
    model_flags = nn.getFlags();
    print('model dir: ' + str(model_flags.model_dir))
    # Train and evaluate the model every `flags.epochs_between_evals` epochs.
    def train_input_fn():
        return nn.input_fn(model_flags.epochs_between_evals, True, model_flags.batch_size, 'train')

    def eval_input_fn():
        return nn.input_fn(1, False, model_flags.batch_size, 'eval')

    run_params = {
      'batch_size':     model_flags.batch_size,
      'train_epochs':   model_flags.train_epochs,
      'model_type':     'deep',
    }

    benchmark_logger = logger.config_benchmark_logger(model_flags)
    benchmark_logger.log_run_info('deep', 'Readmission Patient', run_params)

    # Train and evaluate the model every `flags.epochs_between_evals` epochs.
    for n in range(model_flags.train_epochs // model_flags.epochs_between_evals):
        model.train(input_fn=train_input_fn)
        results = model.evaluate(input_fn=eval_input_fn)
        # Display evaluation metrics
        tf.logging.info('Results at epoch %d / %d',
                        (n + 1) * model_flags.epochs_between_evals,
                        model_flags.train_epochs)
        tf.logging.info('-' * 60)

        for key in sorted(results):
            tf.logging.info('%s: %s' % (key, results[key]))

        benchmark_logger.log_evaluation_result(results)

        if model_helpers.past_stop_threshold(model_flags.stop_threshold, results['accuracy']):
            break;

        # Export the model
        print('export the model?')
        if n%10==0 and model_flags.export_dir is not None:
            nn.export_model()


def main(_):
    run_deep(flags.FLAGS)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    define_flags()
    absl_app.run(main)
