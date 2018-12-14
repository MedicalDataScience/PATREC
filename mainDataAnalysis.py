
import os

from analyzing.DataAnalyzer import DataAnalyzer
from utils.Dataset import Dataset
from utils.DatasetOptions import DatasetOptions

import helpers.constants as constants
import helpers.constantsNZ as constantsNZ

dirProject = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
dirData = dirProject + 'data/';
dirPlotsBase = dirProject + 'plots/feature_comparison_wiederkehrer_normal/'


dict_options_analyzing = {
    'dir_data':             dirData,
    'data_prefix':          'patrec',
    'dataset':              '20122015',
    'encoding':             'categorical',
    'newfeatures':          {'names': constants.NEW_FEATURES},
    'featurereduction':     None,
    'grouping':             'verylightgrouping',
    'filtering':            'cardiovascular'
}

options = DatasetOptions(dict_options_analyzing);
dataset = Dataset(options);

if options.getOptionsFiltering() is not None:
    dirPlots = dirPlotsBase + options.getOptionsFiltering() + '/';
else:
    dirPlots = dirPlotsBase;

if not os.path.exists(dirPlots):
    os.makedirs(dirPlots);

analyzer = DataAnalyzer(options, dirPlots)
analyzer.doFeatureComparison()
# analyzer.checkWiederkehrer();

# avg_num_subgrp = analyzer.getAvgNumberSubgroup('DK')
# print('avg num DK: ' + str(avg_num_subgrp))

# analyzer.getNumberHauptdiagnose();
# analyzer.getNumberColumnsSubgroup('DK');
# analyzer.getNumberColumnsSubgroup('CHOP');
# analyzer.getNumberColumnsSubgroup('OE');