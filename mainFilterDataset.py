
import os

from utils.DatasetFilter import DatasetFilter
from utils.Dataset import Dataset
from utils.DatasetOptions import DatasetOptions

import helpers.constants as constants
import helpers.constantsNZ as constantsNZ

dirProject = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
dirData = dirProject + 'data/';
dirPlotsBase = dirProject + 'plots/feature_comparison_wiederkehrer_normal/'


dict_options_analyzing = {
    'dir_data':                 dirData,
    'data_prefix':              'patrec',
    'dataset':                  '20122015',
    'grouping':                 'verylightgrouping',
    'encoding':                 'categorical',
    'newfeatures':              {'names': constants.NEW_FEATURES},
    'featurereduction':         None,
    'filter_options':           'chronic_lung'
}

options = DatasetOptions(dict_options_analyzing);
dataset = Dataset(options);

datafilter = DatasetFilter(options);
datafilter.filterDataDisease()
