
import sys
import pandas as pd

from helpers.helpers import getDKlightGrouping
from helpers.helpers import getOEgrouping
from helpers.helpers import getDRGgrouping
from helpers.helpers import getAufnahmeartValues
from helpers.helpers import getEntlassartValues
from helpers.helpers import getEintrittsartValues
from helpers.helpers import getEntlassBereichValues
from helpers.helpers import getVersicherungsklasseValues
from helpers.helpers import getForschungskonsentValues
from helpers.helpers import getMonthValues
from helpers.helpers import getDayValues
from helpers.helpers import getYearValues
from helpers.helpers import getLiegestatusValues
from helpers.helpers import getGeschlechtValues
from helpers.helpers import getFeaturesToCategorize
from helpers.helpers import getCountFeaturesToBinarize
from helpers.helpers import getVerweildauerCategories
from helpers.helpers import getAlterCategories
from helpers.helpers import getLOSState
from helpers.helpers import getAgeState


#   categorical feature have to be encoded properly if fed to a ML method
#   In Tensorflow, you can feed the un-encoded categorical features, for 'normal'
#   methods, the encoding has to be done beforehands --> here :)
class FeatureEncoder:

    def __init__(self, options_dataset):
        self.options = options_dataset;
        self.filename_options_in = self.options.getFeatureSet();
        return;

    def __prepareHauptdiagnose(self, valStr):
        return valStr[:2];

    def __prepareMDC(self, valStr):
        return str(valStr)[0];

    def __categorizeValues(self, df, featurename, group_values):
        new_headers = [];
        print('group values: ' + str(group_values))
        for g in group_values:
            h_new = featurename + '_' + str(g);
            new_headers.append(h_new);
        print('len(categorical_headers): ' + str(len(new_headers)))
        print(new_headers)
        df_new = pd.DataFrame(index=df.index, columns=new_headers);
        df_new = df_new.fillna(0);
        for index, row in df.iterrows():
            val = str(row[featurename]);
            assert (val in group_values), "The current value is not in the list of possible value for this feature: %s" % str(val)
            col_new = featurename + '_' + str(val);
            df_new.at[index, col_new] = 1;
        return df_new;


    def __categorizeBinaryValues(self, df, featurename, group_values):
        new_headers = [];
        new_headers.append(featurename + '_' + str(group_values[0]));
        print('group values: ' + str(group_values))
        print('len(categorical_headers): ' + str(len(new_headers)))
        print(new_headers)
        df_new = pd.DataFrame(index=df.index, columns=new_headers);
        df_new = df_new.fillna(0);
        for index, row in df.iterrows():
            val = str(row[featurename]);
            assert (val in group_values), "The current value is not in the list of possible value for this feature"
            col_new = featurename + '_' + str(val);
            if val == group_values[0]:
                df_new.at[index, col_new] = 1;
            else:
                df_new.at[index, col_new] = 0;
        return df_new;


    def __encodeCategoricalFeature(self, df, name):
        print('encode feature: ' + str(name));
        columns = list(df.columns);
        if name in columns:
            if name == 'Hauptdiagnose':
                df_categorical = self.__categorizeValues(df, name, getDKlightGrouping());
            elif name == 'AufnehmOE':
                df_categorical = self.__categorizeValues(df, name, getOEgrouping());
            elif name == 'EntlassOE':
                df_categorical = self.__categorizeValues(df, name, getOEgrouping());
            elif name == 'DRGCode':
                df_categorical = self.__categorizeValues(df, name, getDRGgrouping())
            elif name == 'Aufnahmeart':
                df_categorical = self.__categorizeValues(df, name, getAufnahmeartValues())
            elif name == 'Entlassart':
                df_categorical = self.__categorizeValues(df, name, getEntlassartValues())
            elif name == 'Eintrittsart':
                df_categorical = self.__categorizeValues(df, name, getEintrittsartValues())
            elif name == 'EntlassBereich':
                df_categorical = self.__categorizeValues(df, name, getEntlassBereichValues())
            elif name == 'Versicherungsklasse':
                df_categorical = self.__categorizeValues(df, name, getVersicherungsklasseValues())
            elif name == 'Entlassmonat' or name == 'Aufnahmemonat':
                df_categorical = self.__categorizeValues(df, name, getMonthValues())
            elif name == 'Aufnahmetag' or name == 'Entlasstag':
                df_categorical = self.__categorizeValues(df, name, getDayValues())
            elif name == 'Entlassjahr' or name == 'Aufnahmejahr':
                df_categorical = self.__categorizeValues(df, name, getYearValues())
            elif name == 'Liegestatus':
                df_categorical = self.__categorizeValues(df, name, getLiegestatusValues())
            elif name == 'Geschlecht':
                df_categorical = self.__categorizeBinaryValues(df, name, getGeschlechtValues())
            elif name == 'Forschungskonsent':
                df_categorical = self.__categorizeBinaryValues(df, name, getForschungskonsentValues())
            else:
                print('this feature is not a categorical one or not known as one: ' + str(name));
                sys.exit();
            print('df_categorical.shape: ' + str(df_categorical.shape))
            df = pd.concat([df, df_categorical], axis=1);
            df = df.drop(name, axis=1);
            print('df.shape: ' + str(df.shape))
        return df;


    def __encodeBinarizableFeature(self, df, name):
        print('binarize feature: ' + str(name));
        columns = list(df.columns);
        if name in columns:
            if name == 'Verweildauer':
                df[name] = df[name].apply(getLOSState);
                df_binary = self.__categorizeValues(df, name, getVerweildauerCategories());
            elif name == 'Eintrittsalter':
                df[name] = df[name].apply(getAgeState);
                df_binary = self.__categorizeValues(df, name, getAlterCategories());
            else:
                print('this feature is not binarizable or the method to binarize it is not yet implemented: ' + str(name));
                sys.exit();
            df = pd.concat([df, df_binary], axis=1);
            df = df.drop(name, axis=1);
            print('df.shape: ' + str(df.shape))
        return df;


    # 'simplify' categorical features with too many variables --> not enough data to 'learn' useful representations
    # from all categories, hence this simplification (at least for now)
    def __preprocessFeatureEncoding(self, df):
        headers_data = list(df.columns);
        num_headers_data = len(headers_data);
        print('num headers: ' + str(num_headers_data))
        print(headers_data)
        df = df.fillna(0);
        for h in headers_data:
            if h == 'Hauptdiagnose':
                df[h] = df[h].apply(self.__prepareHauptdiagnose);
            elif h == 'AufnehmOE':
                df[h] = df[h];
            elif h == 'EntlassOE':
                df[h] = df[h];
            elif h == 'DRGCode':
                df[h] = df['DRGCode'].apply(self.__prepareMDC);
        return df;


    def encodeFeatures(self):
        encoding = self.options.getEncodingScheme();
        assert(encoding is not None, 'an encoding algorithm has to be selected..exit');
        assert(encoding in ['categorical', 'binary', 'embedding'], 'feature encoding scheme is not known...please select one of the following: categorical, binary, embedding');

        dir_data = self.options.getDirData();
        dataset = self.options.getDatasetName();
        print('encode features: ' + str(encoding))
        strFilename_in = dataset + '_REST_' + self.filename_options_in;
        strFilename_out = strFilename_in + '_' + encoding;
        filename_data_in = dir_data + 'data_' + strFilename_in + '.csv';
        filename_data_out = dir_data + 'data_' + strFilename_out + '.csv';

        df = pd.read_csv(filename_data_in);
        df = self.__preprocessFeatureEncoding(df);

        if encoding == 'categorical' or encoding == 'binary':
            categorical_features = getFeaturesToCategorize();
            for feat in categorical_features:
                df = self.__encodeCategoricalFeature(df,feat);
            if encoding == 'binary':
                binarizable_features = getCountFeaturesToBinarize();
                for bin_feat in binarizable_features:
                    df = self.__encodeBinarizableFeature(df, bin_feat);
        else:
            print('encoding scheme is not known... no encoding applied')

        print('encoded: df.shape: ' + str(df.shape))
        df.to_csv(filename_data_out, line_terminator='\n', index=False);

