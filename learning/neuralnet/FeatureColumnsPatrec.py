
import tensorflow as tf
import math as math

import helpers.helpers as helpers

class FeatureColumnsPatrec:
    def __init__(self, dataset_options):
        self.dataset_options = dataset_options;
        return;

    def buildModelColumns(self):
        """Builds a set of wide and deep feature columns."""
        # Continuous columns

        geschlecht = tf.feature_column.categorical_column_with_vocabulary_list(
            'Geschlecht', self.dataset_options.getFeatureCategories('Geschlecht')
        )
        versicherungsklasse = tf.feature_column.categorical_column_with_vocabulary_list(
            'Versicherungsklasse', self.dataset_options.getFeatureCategories('Versicherungsklasse')
        )
        forschungskonsent = tf.feature_column.categorical_column_with_vocabulary_list(
            'Forschungskonsent', self.dataset_options.getFeatureCategories('Forschungskonsent')
        )
        entlassbereich = tf.feature_column.categorical_column_with_vocabulary_list(
            'EntlassBereich', self.dataset_options.getFeatureCategories('EntlassBereich')
        )
        aufnahmeart = tf.feature_column.categorical_column_with_vocabulary_list(
            'Aufnahmeart', self.dataset_options.getFeatureCategories('Aufnahmeart')
        )
        aufnahmemonat = tf.feature_column.categorical_column_with_vocabulary_list(
            'Aufnahmemonat', self.dataset_options.getFeatureCategories('Aufnahmemonat')
        )
        entlassmonat = tf.feature_column.categorical_column_with_vocabulary_list(
            'Entlassmonat', self.dataset_options.getFeatureCategories('Entlassmonat')
        )
        aufnahmetag = tf.feature_column.categorical_column_with_vocabulary_list(
            'Aufnahmetag', self.dataset_options.getFeatureCategories('Aufnahmetag')
        )
        entlasstag = tf.feature_column.categorical_column_with_vocabulary_list(
            'Entlasstag', self.dataset_options.getFeatureCategories('Entlasstag')
        )
        liegestatus = tf.feature_column.categorical_column_with_vocabulary_list(
            'Liegestatus', self.dataset_options.getFeatureCategories('Liegestatus')
        )
        entlassart = tf.feature_column.categorical_column_with_vocabulary_list(
            'Entlassart', self.dataset_options.getFeatureCategories('Entlassart')
        )
        eintrittsart = tf.feature_column.categorical_column_with_vocabulary_list(
            'Eintrittsart', self.dataset_options.getFeatureCategories('Eintrittsart')
        )
        aufnahmejahr = tf.feature_column.categorical_column_with_vocabulary_list(
            'Aufnahmejahr', self.dataset_options.getFeatureCategories('Aufnahmejahr')
        )
        entlassjahr = tf.feature_column.categorical_column_with_vocabulary_list(
            'Entlassjahr', self.dataset_options.getFeatureCategories('Entlassjahr')
        )

        eintrittsalter = tf.feature_column.numeric_column('Eintrittsalter', dtype=tf.float32)
        verweildauer = tf.feature_column.numeric_column('Verweildauer', dtype=tf.float32);

        langlieger = tf.feature_column.numeric_column('Langlieger', dtype=tf.int64);
        dksepsis = tf.feature_column.numeric_column('DKSepsis_1', dtype=tf.int64);
        oeintensiv = tf.feature_column.numeric_column('OEIntensiv_1', dtype=tf.int64);

        categories_hauptdiagnose = self.dataset_options.getFeatureCategories('Hauptdiagnose');
        hauptdiagnose = tf.feature_column.categorical_column_with_vocabulary_list(
            'Hauptdiagnose', categories_hauptdiagnose
        )
        oe_aufnehm = tf.feature_column.categorical_column_with_vocabulary_list(
            'AufnehmOE', self.dataset_options.getFeatureCategories('AufnehmOE')
        )
        oe_entlass = tf.feature_column.categorical_column_with_vocabulary_list(
            'EntlassOE', self.dataset_options.getFeatureCategories('EntlassOE')
        )
        mdc = tf.feature_column.categorical_column_with_vocabulary_list(
            'DRGCode', self.dataset_options.getFeatureCategories('DRGCode')
        )
        procedures = tf.feature_column.categorical_column_with_vocabulary_list(
            'CHOP', helpers.getCHOPgrouping()
        )
        categories_nebendiagnose = self.dataset_options.getFeatureCategories('Hauptdiagnose');
        embedding_dim_nebendiagnose = int(round(math.sqrt(math.sqrt(len(categories_nebendiagnose)))));
        nebendiagnose = tf.feature_column.categorical_column_with_vocabulary_list(
            'DK', categories_nebendiagnose
        )
        oe = tf.feature_column.categorical_column_with_vocabulary_list(
            'OE', helpers.getOEgrouping()
        )

        embedding_dim_hauptdiagnose = int(round(math.sqrt(math.sqrt(len(categories_hauptdiagnose)))));
        print('embedding dim hauptdiagnose: ' + str(embedding_dim_hauptdiagnose))

        feature_columns = []
        feature_columns.append(eintrittsalter);
        feature_columns.append(verweildauer);
        feature_columns.append(tf.feature_column.indicator_column(geschlecht));
        feature_columns.append(tf.feature_column.indicator_column(forschungskonsent));
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=versicherungsklasse, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=entlassbereich, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=aufnahmeart, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=aufnahmejahr, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=entlassjahr, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=aufnahmemonat, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=entlassmonat, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=aufnahmetag, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=entlasstag, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=liegestatus, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=entlassart, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=eintrittsart, dimension=2))
        feature_columns.append(
            tf.feature_column.embedding_column(categorical_column=hauptdiagnose, dimension=embedding_dim_hauptdiagnose))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=oe_aufnehm, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=oe_entlass, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=mdc, dimension=2))
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=procedures, dimension=9));
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=nebendiagnose, dimension=embedding_dim_nebendiagnose));
        feature_columns.append(tf.feature_column.embedding_column(categorical_column=oe, dimension=2));
        feature_columns.append(langlieger);
        feature_columns.append(dksepsis);
        feature_columns.append(oeintensiv);

        print('len(feature_columns): ' + str(len(feature_columns)));
        return feature_columns;

    def getDefaultValues(self, headers):
        default_values = [];
        for h in headers:
            if h == 'Aufnahmeart':
                default_values.append(['']);
            elif h == 'Eintrittsalter':
                default_values.append([0.0]);
            elif h == 'Entlassart':
                default_values.append(['']);
            elif h == 'EntlassBereich':
                default_values.append(['']);
            elif h == 'Versicherungsklasse':
                default_values.append(['']);
            elif h == 'Forschungskonsent':
                default_values.append(['']);
            elif h == 'Geschlecht':
                default_values.append(['']);
            elif h == 'Verweildauer':
                default_values.append([0.0]);
            elif h == 'DKSepsis_1':
                default_values.append([0.0]);
            elif h == 'OEIntensiv_1':
                default_values.append([0.0]);
            elif h == 'Wiederkehrer':
                default_values.append([0.0])
            elif h == 'Aufnahmemonat':
                default_values.append([''])
            elif h == 'Entlassmonat':
                default_values.append([''])
            elif h == 'Aufnahmetag':
                default_values.append([''])
            elif h == 'Entlasstag':
                default_values.append([''])
            elif h == 'Aufnahmejahr':
                default_values.append([''])
            elif h == 'Entlassjahr':
                default_values.append([''])
            elif h == 'Patient':
                default_values.append([0.0])
            elif h == 'Aufnahmedatum':
                default_values.append([0.0])
            elif h == 'Entlassdatum':
                default_values.append([0.0])
            elif h == 'Liegestatus':
                default_values.append([''])
            elif h == 'Langlieger':
                default_values.append([0.0])
            elif h == 'Eintrittsart':
                default_values.append([''])
            elif h == 'Hauptdiagnose':
                default_values.append([''])
            elif h == 'AufnehmOE':
                default_values.append([''])
            elif h == 'EntlassOE':
                default_values.append([''])
            elif h == 'DRGCode':
                default_values.append([''])
            elif h == 'DK':
                default_values.append([''])
            elif h == 'OE':
                default_values.append([''])
            elif h == 'CHOP':
                default_values.append([''])
            else:
                default_values.append([''])
        print('len(default_values): ' + str(len(default_values)))
        return default_values;