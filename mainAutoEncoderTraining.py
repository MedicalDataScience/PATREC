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


from learning.neuralnet.AutoEncoderModel import AutoEncoderModel
from learning.neuralnet.FeatureColumnsAutoEncoderNZ import FeatureColumnsAutoEncoderNZ
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
    dict_data_train = {
        'dir_data':             DIRPROJECT + 'data/',
        'data_prefix':          'nz',
        'dataset':              '2012',
        'encoding':             'categorical',
        'newfeatures':          None,
        'featurereduction':     {'method': 'ONLYDIAG'},
        'grouping':             'verylightgrouping'
    }
    dataset_options_train = DatasetOptions(dict_data_train);
    dataset_options_eval = None;

    if dict_data_train['data_prefix'] == 'nz':
        feature_columns = FeatureColumnsAutoEncoderNZ(dataset_options=dataset_options_train);
    else:
        print('unknown data prefix..exit')
        sys.exit()

    dict_dataset_options = {
        'train':    dataset_options_train,
        'eval':     dataset_options_eval,
        'test':     None
    }

    nn = AutoEncoderModel('train', dict_dataset_options, feature_columns, flags_obj);
    nn.train();


def main(_):
    run_deep(flags.FLAGS)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    define_flags()
    absl_app.run(main)
