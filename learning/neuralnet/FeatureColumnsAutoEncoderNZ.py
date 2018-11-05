
import tensorflow as tf
import numpy as np

import helpers.helpers as helpers

class FeatureColumnsAutoEncoderNZ:
    def __init__(self, dataset_options):
        self.dataset_options = dataset_options;
        return;

    def buildModelColumns(self):
        """Builds a set of wide and deep feature columns."""
        # Continuous columns

        diag_other = helpers.getDKgrouping();
        feature_columns = [];
        for d in diag_other:
            d_feat = tf.feature_column.numeric_column('diag_DIAG_' + str(d), dtype=tf.float32)
            feature_columns.append(d_feat);

        print('number of feature columns: ' + str(len(feature_columns)))
        return feature_columns;


    def getDefaultValues(self, headers):
        diag_other = helpers.getDKgrouping();
        num_cols_data = len(diag_other);
        default_values = num_cols_data*[[0.0]];
        return default_values;
