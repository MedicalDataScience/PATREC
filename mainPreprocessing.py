import os

from utils.DatasetOptions import DatasetOptions
from preprocessing.Preprocessor import Preprocessor

import helpers.constants as constantsPATREC


# dirProject = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
dirProject = '/home/thomas/fusessh/scicore/projects/patrec'
dirData = os.path.join(dirProject, 'data');

dict_dataset_options = {
    'dir_data':                 dirData,
    'data_prefix':              'patrec',
    'dataset':                  '20162017',
    # 'subgroups':                ['DK'],
    'grouping':                 'verylightgrouping',
    'encoding':                 'embedding',
    'newfeatures':              None,
    'featurereduction':         None
}

options = DatasetOptions(dict_dataset_options);
preproc = Preprocessor(options);
# preproc.splitColumns();
# preproc.clean()
preproc.group()
preproc.createFeatureSet()
preproc.encodeFeatures();
preproc.fuse();
