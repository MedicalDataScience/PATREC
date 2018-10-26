
import tensorflow as tf
import math as math

import helpers.helpers as helpers

class FeatureColumnsPatrecFusion:
    def __init__(self, dataset_options):
        self.dataset_options = dataset_options;
        return;

    def buildModelColumns(self):
        """Builds a set of wide and deep feature columns."""
        # Continuous columns

        geschlecht = tf.feature_column.categorical_column_with_vocabulary_list(
            'Geschlecht', self.dataset_options.getFeatureCategories('Geschlecht')
        )
        eintrittsalter = tf.feature_column.numeric_column('Eintrittsalter', dtype=tf.float32)
        verweildauer = tf.feature_column.numeric_column('Verweildauer', dtype=tf.float32);

        categories_hauptdiagnose = self.dataset_options.getFeatureCategories('Hauptdiagnose');
        hauptdiagnose = tf.feature_column.categorical_column_with_vocabulary_list(
            'Hauptdiagnose', categories_hauptdiagnose
        )
        nebendiagnose = tf.feature_column.categorical_column_with_vocabulary_list(
            'DK', helpers.getDKverylightGrouping()
        )

        feature_columns = []
        feature_columns.append(eintrittsalter);
        feature_columns.append(verweildauer);
        feature_columns.append(tf.feature_column.indicator_column(geschlecht));
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=hauptdiagnose,
                                                                  dimension=8))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=nebendiagnose, dimension=8));
        return feature_columns;

    def getDefaultValues(self, headers):
        default_values = [];
        for h in headers:
            if h == 'Geschlecht':
                default_values.append(['']);
            elif h == 'Verweildauer':
                default_values.append([0.0]);
            elif h == 'Eintrittsalter':
                default_values.append([0.0]);
            elif h == 'Wiederkehrer':
                default_values.append([0.0])
            elif h == 'Hauptdiagnose':
                default_values.append([''])
            elif h == 'DK':
                default_values.append([''])
            else:
                default_values.append([''])
        return default_values;


    def getEmbeddingLayerNames(self):
        name_main_diag = 'input_layer/Hauptdiagnose_embedding/embedding_weights';
        name_diag = 'input_layer/DK_embedding/embedding_weights';
        return [name_main_diag, name_diag];

    def getConversionDict(self):
        conv_dict = {
            'input_layer/Hauptdiagnose_embedding/embedding_weights': 'input_layer/main_diag_embedding/embedding_weights',
            'input_layer/DK_embedding/embedding_weights': 'input_layer/diag_embedding/embedding_weights',
            # 'input_layer/Geschlecht_embedding/embedding_weights': 'input_layer/gender_embedding/embedding_weights',
            # 'input_layer/Eintrittsalter_numeric': 'input_layer/age_dsch_numeric',
            # 'input_layer/Verweildayer_numeric': 'input_layer/los_dsch_numeric',
        }
        return conv_dict;
