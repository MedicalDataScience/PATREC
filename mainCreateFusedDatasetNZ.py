
import os
import pandas as pd

from utils.DatasetOptions import DatasetOptions
from utils.Dataset import Dataset

import helpers.constantsNZ as constantsNZ

if __name__ == '__main__':
    dirProject = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';
    dirData = dirProject + 'data/';
    dirResultsBase = dirProject + 'results/';
    dirModelsBase = dirProject + 'classifiers/'

    years = range(2001, 2017);

    df_all_years = pd.DataFrame()
    for year in years:
        dict_options_dataset = {
            'dir_data':             dirData,
            'data_prefix':          'nz',
            'dataset':              str(year),
            'encoding':             'embedding',
            'grouping':             'verylightgrouping',
            'newfeatures':          None,
            'featurereduction':     {'method': 'FUSION'}
        }

        options_dataset_year = DatasetOptions(dict_options_dataset);
        dataset_year = Dataset(options_dataset_year);
        df_balanced_year = dataset_year.getBalancedSubSet();
        print(df_balanced_year.shape)
        df_all_years = df_all_years.append(df_balanced_year);


    print('df balanced all years: ' + str(df_all_years.shape))

    encoding = options_dataset_year.getEncodingScheme();
    grouping = options_dataset_year.getGroupingName();
    featureset = options_dataset_year.getFeatureSetStr();
    filename_data_years = dirData + 'data_nz_' + str(min(years)) + str(max(years)) + '_' + featureset + '_' + encoding + '_' + grouping + '.csv';
    df_all_years.to_csv(filename_data_years, line_terminator='\n', index=False);


