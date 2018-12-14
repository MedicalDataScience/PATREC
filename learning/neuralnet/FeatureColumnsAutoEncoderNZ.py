
import tensorflow as tf
import numpy as np

import helpers.helpers as helpers

class FeatureColumnsAutoEncoderNZ:
    def __init__(self, dataset_options):
        self.dataset_options = dataset_options;
        return;

    # def buildModelColumns(self):
    #     """Builds a set of wide and deep feature columns."""
    #     # Continuous columns
    #     diag_other = self.dataset_options.getDiagGroupNames();
    #     feature_columns = [];
    #     for d in diag_other:
    #         d_feat = tf.feature_column.numeric_column(str(d), dtype=tf.float32)
    #         feature_columns.append(d_feat);
    #
    #     # print(feature_columns)
    #     print('number of feature columns: ' + str(len(feature_columns)))
    #     return feature_columns;

    def buildModelColumns(self):
        """Builds a set of wide and deep feature columns."""
        # Continuous columns

        diag_other = self.dataset_options.getDiagGroupNames();
        other_diag = tf.feature_column.categorical_column_with_vocabulary_list(
            'diag', diag_other
        )

        feature_columns = []
        feature_columns.append(tf.feature_column.indicator_column(other_diag))
        return feature_columns;


    def getDefaultValues(self, headers):
        default_values = [];
        for h in headers:
            if h == 'age_dsch':
                default_values.append([0]);
            elif h == 'gender':
                default_values.append(['']);
            elif h == 'los':
                default_values.append([0]);
            elif h == 'early_readmission_flag':
                default_values.append([0])
            elif h == 'main_diag_ind':
                default_values.append([0])
            elif h == 'main_diag':
                default_values.append([''])
            elif h == 'diag':
                default_values.append([''])
            else:
                default_values.append([''])
        return default_values;


    def getConversionDict(self):
        conv_dict = None
        return conv_dict;
