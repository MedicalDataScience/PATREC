
import os

from analyzing.DataAnalyzer import DataAnalyzer
from utils.Dataset import Dataset
from utils.DatasetOptions import DatasetOptions

dirProject = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
dirData = dirProject + 'data/';
dirPlotsBase = dirProject + 'plots/feature_comparison_wiederkehrer_normal/'


new_features = ['previous_visits', 'ratio_los_age', 'ratio_numDK_age', 'ratio_los_numDK', 'ratio_numCHOP_age', 'ratio_los_numOE', 'ratio_numOE_age', 'mult_los_numCHOP', 'mult_equalOE_numDK', 'diff_alos'];
# new_features = ['diff_alos']
options_standard = None;
options_newfeatures = {'subgroups': ['CHOP', 'DK', 'OE'], 'names_new_features': new_features};
options_reduction = {'reduction_method': 'NOADMIN'};
options_featureset = None;
options_grouping = None;
options_encoding = None;
options_filtering = 'Hauptdiagnose_I2';

dict_options_analyzing = {
    'dir_data':             dirData,
    'dataset':              '20122015',
    'subgroups':            ['OE', 'DK', 'CHOP'],
    'featureset':           'newfeatures',
    'options_featureset':   options_featureset,
    'encoding':             'categorical',
    'options_encoding':     options_encoding,
    'grouping':             'grouping',
    'options_grouping':     options_grouping,
    'filtering':            options_filtering,
}

options = DatasetOptions(dict_options_analyzing);
dataset = Dataset(options);

if options_filtering is not None:
    dirPlots = dirPlotsBase + options_filtering + '/';
else:
    dirPlots = dirPlotsBase;

if not os.path.exists(dirPlots):
    os.makedirs(dirPlots);

analyzer = DataAnalyzer(dataset, dirPlots)
analyzer.doFeatureComparison()