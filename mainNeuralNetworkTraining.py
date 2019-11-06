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

DIRPROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DIRPROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
# DIRPROJECT = DIRPROJECT.replace("\\", "/")

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

import helpers.constants as constantsPATREC

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

    model_dir = os.path.join(DIRPROJECT, "patients_model")
    export_dir = os.path.join(model_dir, "export_model")

    flags_core.set_defaults(data_dir=DIRPROJECT + 'data/',
                            model_dir=model_dir,
                            export_dir=export_dir,
                            hidden_units=[60, 40, 40, 20],
                            train_epochs=1000,
                            epochs_between_evals=1,
                            batch_size=64,
                            learningrate=0.001)

    flags.DEFINE_bool('enable_dp', True, 'Enable Differential Privacy')
    flags.DEFINE_float('dp_eps', 10, 'Differential Privacy Epsilon')
    flags.DEFINE_float('dp_delta', 1e-5, 'Differential Privacy Delta')
    flags.DEFINE_float('dp_sigma', 0.5, 'Differential Privacy Noise Amount')
    flags.DEFINE_float('dp_c', 1, 'Differential Privacy Norm Clipping Amount')
    flags.DEFINE_integer('dp_num_microbatches', 64, 'Number of microbatches to use in DP optimizer')
    flags.DEFINE_bool('force_cpu', False, 'Force CPU usage')


def run_deep(flags_obj):
    """Run Wide-Deep training and eval loop.
    Args:
    flags_obj: An object containing parsed flag values.
    """

    # dirProject = '/home/thomas/fusessh/scicore/projects/patrec/projects/PATREC'
    dirProject = "Z:\\projects\\PATREC"
    dirData = os.path.join(dirProject, 'data');
    dict_options_dataset_training = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20122015',
        'grouping':         'verylightgrouping',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': None,
        'filtering':        None,
        'balanced':         False,
        'resample':         True
    }
    dataset_options_train = DatasetOptions(dict_options_dataset_training);

    dataset_options_eval = None;


    if dict_options_dataset_training['data_prefix'] == 'nz':
        feature_columns_nz_fusion = FeatureColumnsNZ(dataset_options=dataset_options_train);
        feature_columns = feature_columns_nz_fusion;
    elif dict_options_dataset_training['data_prefix'] == 'patrec':
        feature_columns_patrec_fusion = FeatureColumnsPatrec(dataset_options=dataset_options_train);
        feature_columns = feature_columns_patrec_fusion;
    else:
        print('unknown data prefix..exit')
        sys.exit()

    dict_dataset_options = {
        'train':        dataset_options_train,
        'eval':         dataset_options_eval,
        'test':         None
    }

    nn = NeuralNetModel('train', dict_dataset_options, feature_columns, flags_obj);
    print(flags_obj.log_dir)
    nn.train();


def main(_):
    run_deep(flags.FLAGS)


if __name__ == '__main__':

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    define_flags()
    absl_app.run(main)
