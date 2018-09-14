

from preprocessing.Preprocessor import Preprocessor

new_features = ['previous_visits', 'ratio_los_age', 'ratio_numDK_age', 'ratio_los_numDK', 'ratio_numCHOP_age',
                'ratio_los_numOE', 'ratio_numOE_age', 'mult_los_numCHOP', 'mult_equalOE_numDK',
                'diff_drg_alos', 'diff_drg_lowerbound', 'diff_drg_upperbound',
                'rel_diff_drg_alos', 'rel_diff_drg_lowerbound', 'rel_diff_drg_upperbound',
                'alos', 'ratio_drg_los_alos'];

# new_features = ['rel_diff_drg_alos', 'rel_diff_drg_lowerbound', 'rel_diff_drg_upperbound']
options_standard = None;
options_newfeatures = {'subgroups': ['CHOP', 'DK', 'OE'], 'names_new_features': new_features};
options_reduction = {'reduction_method': 'NOADMIN'};

options_preprocessing = {
    'dataset':              '20122015',
    'subgroups':            ['OE', 'DK', 'CHOP'],
    'featureset':           'newfeatures',
    'options_featureset':   options_newfeatures,
    'encoding':             'categorical',
    'grouping':             'grouping',
    'options_grouping':     None,
    'chunksize':            10000,
}


preproc = Preprocessor(options_preprocessing);
# preproc.splitColumns();
# preproc.clean()
# preproc.group(filename_options_out='grouping')      #, names_newfeatures=['numCHOP', 'numDK', 'numOE']
# preproc.prepare(filename_options_in='grouping', filename_options_out='normal_grouping')


# preproc.addFeatures(filename_options_in='normal_grouping', filename_options_out='newfeatures', names_additional_features=new_features)

# preproc.reduceFeatures(filename_options_in='normal', feature_reduction_method=None);

# preproc.encodeFeatures(filename_options_in='newfeatures', encoding='categorical');

# preproc.splitDatasetIntoTrainingTestingSet(filename_options_in='embedding_grouping')

# preproc.createFeatureSet()
# preproc.encodeFeatures();
# preproc.fuse();
preproc.filterData('Hauptdiagnose_I2');
