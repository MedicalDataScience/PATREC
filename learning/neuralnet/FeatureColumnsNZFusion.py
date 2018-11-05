
import tensorflow as tf
import math as math

import helpers.helpers as helpers

class FeatureColumnsNZFusion:
    def __init__(self, dataset_options):
        self.dataset_options = dataset_options;
        return;

    def buildModelColumns(self):
        """Builds a set of wide and deep feature columns."""
        # Continuous columns

        gender = tf.feature_column.categorical_column_with_vocabulary_list(
            'gender', self.dataset_options.getFeatureCategories('gender')
        )

        main_diag = tf.feature_column.categorical_column_with_vocabulary_list(
            'main_diag', self.dataset_options.getFeatureCategories('main_diag')
        )

        age = tf.feature_column.numeric_column('age_dsch', dtype=tf.float32)
        los = tf.feature_column.numeric_column('los', dtype=tf.float32);

        diag_other = helpers.getDKverylightGrouping()
        other_diag = tf.feature_column.categorical_column_with_vocabulary_list(
            'diag', diag_other
        )

        # feature_columns = []
        feature_columns = tf.feature_column.shared_embedding_columns([main_diag, other_diag], dimension=128)
        feature_columns.append(age);
        feature_columns.append(los);
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=gender, dimension=1))
        print('len(feature_columns): ' + str(len(feature_columns)))
        # feature_columns.append(tf.feature_column.embedding_column(categorical_column=main_diag, dimension=26))
        # feature_columns.append(tf.feature_column.embedding_column(categorical_column=other_diag,
        #                                                           dimension=26,
        #                                                           combiner='sqrtn'));
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
            elif h == 'main_diag':
                default_values.append([''])
            elif h == 'diag':
                default_values.append([''])
            else:
                default_values.append([''])
        return default_values;


    def getEmbeddingLayerNames(self):
        name_main_diag = 'input_layer/main_diag_embedding/embedding_weights';
        name_diag = 'input_layer/diag_embedding/embedding_weights';
        return [name_main_diag, name_diag];


    def getConversionDict(self):
        conv_dict = None
        return conv_dict;