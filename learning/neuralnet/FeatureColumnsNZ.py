
import tensorflow as tf
import math as math

import helpers.helpers as helpers

class FeatureColumnsNZ:
    def __init__(self, dataset_options):
        self.dataset_options = dataset_options;
        return;

    def buildModelColumns(self):
        """Builds a set of wide and deep feature columns."""
        # Continuous columns

        gender = tf.feature_column.categorical_column_with_vocabulary_list(
            'gender', self.dataset_options.getFeatureCategories('gender')
        )
        adm_src = tf.feature_column.categorical_column_with_vocabulary_list(
            'adm_src', self.dataset_options.getFeatureCategories('adm_src')
        )
        adm_type = tf.feature_column.categorical_column_with_vocabulary_list(
            'adm_type', self.dataset_options.getFeatureCategories('adm_type')
        )
        event_type = tf.feature_column.categorical_column_with_vocabulary_list(
            'event_type', self.dataset_options.getFeatureCategories('event_type')
        )
        end_type = tf.feature_column.categorical_column_with_vocabulary_list(
            'end_type', self.dataset_options.getFeatureCategories('end_type')
        )
        facility_type = tf.feature_column.categorical_column_with_vocabulary_list(
            'facility_type', self.dataset_options.getFeatureCategories('facility_type')
        )
        agency_type = tf.feature_column.categorical_column_with_vocabulary_list(
            'agency_type', self.dataset_options.getFeatureCategories('agency_type')
        )
        private_flag = tf.feature_column.categorical_column_with_vocabulary_list(
            'private_flag', self.dataset_options.getFeatureCategories('private_flag')
        )
        purchaser = tf.feature_column.categorical_column_with_vocabulary_list(
            'purchaser', self.dataset_options.getFeatureCategories('purchaser')
        )
        short_stay_flag = tf.feature_column.categorical_column_with_vocabulary_list(
            'Short_Stay_ED_Flag', self.dataset_options.getFeatureCategories('Short_Stay_ED_Flag')
        )
        transfer_event_flag = tf.feature_column.categorical_column_with_vocabulary_list(
            'transfer_event_flag', self.dataset_options.getFeatureCategories('transfer_event_flag')
        )
        main_diag = tf.feature_column.categorical_column_with_vocabulary_list(
            'main_diag', self.dataset_options.getFeatureCategories('main_diag')
        )

        age = tf.feature_column.numeric_column('age_dsch', dtype=tf.float32)
        los = tf.feature_column.numeric_column('los', dtype=tf.float32);

        diag_other = ['DIAG_' + dk for dk in helpers.getDKgrouping()]
        other_diag = tf.feature_column.categorical_column_with_vocabulary_list(
            'diag', diag_other
        )

        feature_columns = []
        feature_columns.append(age);
        feature_columns.append(los);
        feature_columns.append(tf.feature_column.indicator_column(adm_src));
        feature_columns.append(tf.feature_column.indicator_column(private_flag));
        feature_columns.append(tf.feature_column.indicator_column(short_stay_flag));
        feature_columns.append(tf.feature_column.indicator_column(transfer_event_flag));
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=gender, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=event_type, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=end_type, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=facility_type, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=agency_type, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=purchaser, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=main_diag, dimension=4))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=other_diag, dimension=2));

        print('len(feature_columns): ' + str(len(feature_columns)));
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
            elif h == 'adm_src':
                default_values.append(['']);
            elif h == 'adm_type':
                default_values.append(['']);
            elif h == 'event_type':
                default_values.append(['']);
            elif h == 'end_type':
                default_values.append(['']);
            elif h == 'facility_type':
                default_values.append(['']);
            elif h == 'agency_type':
                default_values.append(['']);
            elif h == 'private_flag':
                default_values.append(['']);
            elif h == 'purchaser':
                default_values.append([''])
            elif h == 'Short_Stay_ED_Flag':
                default_values.append([''])
            elif h == 'early_readmission_flag':
                default_values.append([0])
            elif h == 'transfer_event_flag':
                default_values.append([''])
            elif h == 'main_diag':
                default_values.append([''])
            elif h == 'diag':
                default_values.append([''])
            else:
                default_values.append([''])
        print('len(default_values): ' + str(len(default_values)))
        return default_values;