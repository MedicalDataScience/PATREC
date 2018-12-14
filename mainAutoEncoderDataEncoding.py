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

DIRPROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';

from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers

from learning.neuralnet.AutoEncoderModel import AutoEncoderModel
from learning.neuralnet.FeatureColumnsAutoEncoderNZ import FeatureColumnsAutoEncoderNZ
from learning.ClassifierNN import OptionsNN
from learning.ClassifierNN import ClassifierNN

from utils.Results import ResultsSingleRun
from utils.Results import Results
from utils.DatasetOptions import DatasetOptions
from utils.Dataset import Dataset

# hidden_units = [100, 100, 50, 50, 25, 25, 25, 25, 10, 10, 10];
# hidden_units = [35, 35, 35, 35, 35, 35, 35, 35, 35, 35];
# hidden_units = [100, 80, 60, 40, 10]
# hidden_units = [20, 20, 20, 10, 10];
# hidden_units = [10, 10, 10, 10, 10];
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

    flags_core.set_defaults(data_dir=DIRPROJECT + 'data/',
                            model_dir='/tmp/patients_model',
                            export_dir='/tmp/patients_model/export_model',
                            train_epochs=250,
                            epochs_between_evals=1,
                            batch_size=160)


def encode(flags_obj):
    """Run Wide-Deep training and eval loop.
    Args:
    flags_obj: An object containing parsed flag values.
    """
    dict_data_training = {
        'dir_data':         DIRPROJECT + 'data/',
        'data_prefix':      'nz',
        'dataset':          '20012016',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': {'method': 'FUSION'},
        'grouping':         'verylightgrouping'
    }
    dataset_options_training = DatasetOptions(dict_data_training);

    dict_data_encoding = {
        'dir_data':         DIRPROJECT + 'data/',
        'data_prefix':      'nz',
        'dataset':          '2017',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': {'method': 'FUSION'},
        'grouping':         'verylightgrouping'
    }
    dataset_options_encoding = DatasetOptions(dict_data_encoding);

    feature_columns = FeatureColumnsAutoEncoderNZ(dataset_options=dataset_options_encoding);

    dict_dataset_options = {
        'train': dataset_options_training,
        'eval': None,
        'test': dataset_options_encoding
    }

    nn = AutoEncoderModel('test', dict_dataset_options, feature_columns, flags_obj);
    diag_encodings = nn.encode();
    print('diag_encodings --> main diag: ' + str(diag_encodings[0].shape))
    print('diag_encodings --> secondary diags: ' + str(diag_encodings[1].shape))

    main_diag_encodings = diag_encodings[0];
    sec_diag_encodings = diag_encodings[1];

    dataset_encoding = Dataset(dataset_options_encoding)
    df_encoding = dataset_encoding.getDf();
    print('df_encoding: ' + str(df_encoding.shape))
    num_encoded_dim = main_diag_encodings.shape[1];

    dir_data = dataset_options_encoding.getDirData();
    dataset = dataset_options_encoding.getDatasetName();
    data_prefix = dataset_options_encoding.getDataPrefix();
    demographic_featurename = dataset_options_encoding.getFilenameOptionDemographicFeatures();
    featureset_str = dataset_options_encoding.getFeatureSetStr();
    encoding = dataset_options_encoding.getEncodingScheme();
    name_event_column = dataset_options_encoding.getEventColumnName();

    name_main_diag = dataset_options_encoding.getNameMainDiag();
    name_sec_diag = dataset_options_encoding.getNameSecDiag();
    df_encoding_sec_diag = df_encoding[name_event_column].to_frame();
    df_encoding_main_diag = df_encoding[name_event_column].to_frame();

    num_encoded_dim = sec_diag_encodings.shape[1];
    for k in range(0, num_encoded_dim):
        new_col_secdiag = name_sec_diag + '_dim_' + str(k);
        df_encoding_sec_diag[new_col_secdiag] = sec_diag_encodings[:,k];

        new_col_maindiag = name_main_diag + '_dim_' + str(k);
        df_encoding_main_diag[new_col_maindiag] = main_diag_encodings[:, k];

    print('df_encoding_main_diag: ' + str(df_encoding_main_diag.shape))
    print('df_encoding_sec_diag: ' + str(df_encoding_sec_diag.shape))

    filename_sec_diag_encoding = dir_data + 'data_' + data_prefix + '_' + dataset + '_' + name_sec_diag + '_' + str(num_encoded_dim) + 'dim.csv';
    filename_main_diag_encoding = dir_data + 'data_' + data_prefix + '_' + dataset + '_' + name_main_diag + '_' + str(num_encoded_dim) + 'dim.csv';

    list_df = [df_encoding_sec_diag[i:i + 10000] for i in range(0, df_encoding_sec_diag.shape[0], 10000)]
    list_df[0].to_csv(filename_sec_diag_encoding, index=False, line_terminator='\n')
    for l in list_df[1:]:
        l.to_csv(filename_sec_diag_encoding, index=False, line_terminator='\n', header=False, mode='a')

    list_df = [df_encoding_main_diag[i:i + 10000] for i in range(0, df_encoding_main_diag.shape[0], 10000)]
    list_df[0].to_csv(filename_main_diag_encoding, index=False, line_terminator='\n')
    for l in list_df[1:]:
        l.to_csv(filename_main_diag_encoding, index=False, line_terminator='\n', header=False, mode='a')




def main(_):
    encode(flags.FLAGS)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_flags()
    absl_app.run(main)
