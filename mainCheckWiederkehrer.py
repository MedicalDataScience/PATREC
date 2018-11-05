
import os

from analyzing.DataAnalyzer import DataAnalyzer
from utils.Dataset import Dataset
from utils.DatasetOptions import DatasetOptions

if __name__ == '__main__':

    dirProject = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
    dirData = dirProject + 'data/';
    dirResultsBase = dirProject + 'results/';
    dirModelsBase = dirProject + 'classifiers/'
    dirPlotsBase = dirProject + 'plots/';

    new_features = ['previous_visits', 'ratio_los_age', 'ratio_numDK_age', 'ratio_los_numDK', 'ratio_numCHOP_age',
                    'ratio_los_numOE', 'ratio_numOE_age', 'mult_los_numCHOP', 'mult_equalOE_numDK',
                    'diff_drg_alos', 'diff_drg_lowerbound', 'diff_drg_upperbound',
                    'rel_diff_drg_alos', 'rel_diff_drg_lowerbound', 'rel_diff_drg_upperbound',
                    'alos', 'ratio_drg_los_alos'];

    # new_features = ['rel_diff_drg_alos', 'rel_diff_drg_lowerbound', 'rel_diff_drg_upperbound']
    options_standard = None;
    options_newfeatures = {'names_new_features': new_features};
    options_reduction = {'reduction_method': 'NOADMIN'};
    options_filtering = None;

    dict_options_dataset = {
        'dir_data':                 dirData,
        'dataset':                  '20122015',
        'subgroups':                ['OE', 'DK', 'CHOP'],
        'featureset':               'newfeatures',
        'options_featureset':       options_newfeatures,
        'grouping':                 'grouping',
        'options_grouping':         None,
        'encoding':                 'categorical',
        'options_encoding':         None,
        'options_filtering':        options_filtering,
        'chunksize':                10000,
        'ratio_training_samples':   0.85,
    }

    options = DatasetOptions(dict_options_dataset);
    dataset = Dataset(options);

    if options_filtering is not None:
        dirPlots = dirPlotsBase + options_filtering + '/';
    else:
        dirPlots = dirPlotsBase;

    if not os.path.exists(dirPlots):
        os.makedirs(dirPlots);

    analyzer = DataAnalyzer(dataset, dirPlots)
    analyzer.checkWiederkehrer()