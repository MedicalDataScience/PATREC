import os
import sys

from preprocessing.PreprocessorNZ import PreprocessorNZ
from utils.DatasetOptions import DatasetOptions

dirProject = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
dirData = dirProject + 'data/';

new_features = ['previous_visits', 'ratio_los_age', 'ratio_numDK_age', 'ratio_los_numDK', 'ratio_numCHOP_age',
                'ratio_los_numOE', 'ratio_numOE_age', 'mult_los_numCHOP', 'mult_equalOE_numDK',
                'diff_drg_alos', 'diff_drg_lowerbound', 'diff_drg_upperbound',
                'rel_diff_drg_alos', 'rel_diff_drg_lowerbound', 'rel_diff_drg_upperbound',
                'alos', 'ratio_drg_los_alos'];

def main(dict_dataset_options):

    options = DatasetOptions(dict_dataset_options);
    preproc = PreprocessorNZ(options);

    print('grouping: ' + str(options.getGroupingName()))
    # preproc.processDischargeFile();
    # preproc.processDiagnosisFile();

    # preproc.createFeatureSet();
    # preproc.encodeFeatures();
    preproc.fuse();




if __name__ == '__main__':
    if len(sys.argv) > 1:
        year_to_process = int(sys.argv[1]);
    else:
        year_to_process = 1988;

    years = [2012]
    # years = [2007, 2008, 2009, 2010, 2011]
    # years = [2001, 2002, 2003, 2004, 2005, 2006]
    for year in years:
        dict_dataset_options = {
            'dir_data':             dirData,
            'data_prefix':          'nz',
            'dataset':              str(year),
            'encoding':             'categorical',
            'grouping':             'verylightgrouping',
            'newfeatures':          None,
            'featurereduction':     {'method': 'ONLYDIAG'}
        }
        print('')
        print('processing year: ' + str(year))
        main(dict_dataset_options)