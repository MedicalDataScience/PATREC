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
import numpy as np
import pandas as pd

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

tf_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models';

print(tf_base_dir)
if not tf_base_dir in sys.path:
    sys.path.append(tf_base_dir);

DIRPROJECT = '/home/thomas/projects/patrec';
# DIRPROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';

from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers

from learning.neuralnet.NeuralNetModel import NeuralNetModel
from learning.neuralnet.FeatureColumnsPatrec import FeatureColumnsPatrec
from learning.neuralnet.FeatureColumnsNZ import FeatureColumnsNZ
from learning.neuralnet.FeatureColumnsNZFusion import FeatureColumnsNZFusion
from learning.neuralnet.FeatureColumnsPatrecFusion import FeatureColumnsPatrecFusion
from learning.ClassifierNN import OptionsNN
from learning.ClassifierNN import ClassifierNN

from utils.Results import ResultsSingleRun
from utils.Results import Results
from utils.DatasetOptions import DatasetOptions


# hidden_units = [100, 100, 50, 50, 25, 25, 25, 25, 10, 10, 10];
# hidden_units = [35, 35, 35, 35, 35, 35, 35, 35, 35, 35];
# hidden_units = [100, 80, 60, 40, 10]
#hidden_units = [20, 20, 20, 10, 10];
#hidden_units = [10, 10, 10, 10, 10];
# hidden_units = [60, 60, 40, 40, 20, 20, 20, 10];
# hidden_units=[60, 40, 20, 10, 10]

# dropout_rate = 0.15;
# batch_norm = True;
# learning_rate = 0.05;



def define_flags():
    """Add supervised learning flags, as well as wide-deep model type."""
    flags_core.define_base()
    flags_core.define_benchmark()

    flags.adopt_module_key_flags(flags_core)

    model_dir = os.path.join(DIRPROJECT, "patients_model")
    export_dir = os.path.join(model_dir, "export_model")

    flags_core.set_defaults(data_dir=os.path.join(DIRPROJECT, 'data'),
                            model_dir=model_dir,
                            export_dir=export_dir,
                            hidden_units=[60, 40, 40, 20],
                            train_epochs=1000,
                            epochs_between_evals=1,
                            batch_size=64,
                            learningrate=0.001)

    flags.DEFINE_bool('enable_dp', False, 'Enable Differential Privacy')
    flags.DEFINE_float('dp_eps', 10, 'Differential Privacy Epsilon')
    flags.DEFINE_float('dp_delta', 1e-5, 'Differential Privacy Delta')
    flags.DEFINE_float('dp_sigma', 0.5, 'Differential Privacy Noise Amount')
    flags.DEFINE_float('dp_c', 1, 'Differential Privacy Norm Clipping Amount')
    flags.DEFINE_integer('dp_num_microbatches', 64, 'Number of microbatches to use in DP optimizer')
    flags.DEFINE_bool('force_cpu', False, 'Force CPU usage')

def predict(flags_obj):
    """Run Wide-Deep training and eval loop.
    Args:
    flags_obj: An object containing parsed flag values.
    """
    dirProject = '/home/thomas/fusessh/scicore/projects/patrec'
    # dirProject = "Z:\\projects\\PATREC"
    dirResultsBase = os.path.join(dirProject, 'results/');
    dirData = os.path.join(dirProject, 'data');

    dict_options_dataset_training = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20122015',
        'grouping':         'verylightgrouping',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': None,
        'filtering':        'EntlassBereich_Gyn',
        'balanced':         False,
        'resample':         False
    }
    dataset_options_training = DatasetOptions(dict_options_dataset_training);

    dict_options_dataset_testing = {
        'dir_data':         dirData,
        'data_prefix':      'patrec',
        'dataset':          '20162017',
        'grouping':         'verylightgrouping',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': None,
        'filtering':        'EntlassBereich_Gyn',
        'balanced':         False,
        'resample':         False
    }
    dataset_options_testing = DatasetOptions(dict_options_dataset_testing);

    if dict_options_dataset_testing['data_prefix'] == 'nz':
        feature_columns = FeatureColumnsNZ(dataset_options=dataset_options_testing);
        # feature_columns_nz_fusion = FeatureColumnsNZFusion(dataset_options=dataset_options_testing);
        # feature_columns = feature_columns_nz_fusion;
    elif dict_options_dataset_testing['data_prefix'] == 'patrec':
        feature_columns = FeatureColumnsPatrec(dataset_options=dataset_options_testing)
        # feature_columns_patrec_fusion = FeatureColumnsPatrecFusion(dataset_options=dataset_options_testing);
        # feature_columns = feature_columns_patrec_fusion;
    else:
        print('unknown data prefix..exit')
        sys.exit()

    dict_dataset_options = {
        'train':        dataset_options_training,
        'eval':         None,
        'test':         dataset_options_testing
    }

    nn = NeuralNetModel('test', dict_dataset_options, feature_columns, flags_obj);
    model_flags = nn.getFlags();

    if model_flags.model_dir.endswith('/'):
        trained_model = model_flags.model_dir.split('/')[-2];
    else:
        trained_model = model_flags.model_dir.split('/')[-1];

    if trained_model.startswith('warmstart'):
        pretrained = 'pretrained';
    else:
        pretrained = None;

    print('warmstart: ' + str(trained_model.startswith('warmstart')))
    print('hidden units: ' + str(model_flags.hidden_units))
    dict_options_nn = {
        'hidden_units':     model_flags.hidden_units,
        'learningrate':     model_flags.learningrate,
        'dropout':          model_flags.dropout,
        'batchnorm':        model_flags.batchnorm,
        'batch_size':       model_flags.batch_size,
        'training_epochs':  model_flags.train_epochs,
        'pretrained':       pretrained,
    }

    options_nn = OptionsNN(model_flags.model_dir, dataset_options_training, options_clf=dict_options_nn);
    classifier_nn = ClassifierNN(options_nn)
    results_all_runs_test = Results(dirResultsBase, dataset_options_training, options_nn, 'test', dataset_options_testing);

    num_runs = 10;
    test_auc = [];
    test_avgprecision = [];
    for k in range(0, num_runs):

        results = nn.predict();
        filename_data_testing = nn.getFilenameDatasetBalanced();
        df_testing_balanced = pd.read_csv(filename_data_testing);

        predictions = [p['probabilities'] for p in results];
        predictions = np.array(predictions);

        print('get labels...: ' + str(filename_data_testing))
        labels = df_testing_balanced[dataset_options_testing.getEarlyReadmissionFlagname()].values;
        res = classifier_nn.setResults(predictions, labels)
        results_all_runs_test.addResultsSingleRun(res);

        auc = res.getAUC();
        avgprecision = res.getAvgPrecision();
        print('')
        print('AUC: ' + str(auc))
        print('avg precision: ' + str(avgprecision))
        print('')
        test_auc.append(auc)
        test_avgprecision.append(avgprecision);


    print('')
    print('mean test auc: ' + str(np.mean(np.array(test_auc))))
    print('mean test avg precision: ' + str(np.mean(np.array(test_avgprecision))))
    print('')
    results_all_runs_test.writeResultsToFileDataset();

    # embedding_names = ['main_diag', 'diag'];
    # for name in embedding_names:
    #     weights = nn.getWeightsEmbeddingLayer(name);
    #     filename_weights = flags_obj.model_dir + '/weights_embedding_' + name + '.npy';
    #     filename_weights_tsv = flags_obj.model_dir + '/weights_embedding_' + name + '.tsv';
    #     np.save(filename_weights, weights);
    #     np.savetxt(filename_weights_tsv, weights, fmt='%1.5f', delimiter='\t', newline='\n');
    


def main(_):
    predict(flags.FLAGS)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    define_flags()
    absl_app.run(main)
