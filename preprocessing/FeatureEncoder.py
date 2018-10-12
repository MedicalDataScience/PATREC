
import sys
import pandas as pd


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

    def __prepareOE(self, valStr):
        return str(int(valStr));


    # def __categorizeBinary(self, df, featurename, feature_values):
    #     new_headers = [];
    #     new_headers.append(featurename + '_' + str(feature_values[0]));
    #     print(new_headers)
    #     df_new = pd.DataFrame(index=df.index, columns=new_headers);
    #     df_new = df_new.fillna(0);
    #     for index, row in df.iterrows():
    #         val = str(row[featurename]);
    #         assert (val in feature_values), "The current value is not in the list of possible value for this feature"
    #         col_new = featurename + '_' + str(val);
    #         if val == feature_values[0]:
    #             df_new.at[index, col_new] = 1;
    #         else:
    #             df_new.at[index, col_new] = 0;
    #     print('df_new: ' + str(df_new.shape))
    #     print(list(df_new.columns))
    #     return df_new;


    def __categorizeMulti(self, df, featurename, feature_values):
        new_headers = [];
        print('group values: ' + str(feature_values))
        for g in feature_values:
            h_new = featurename + '_' + str(g);
            new_headers.append(h_new);
        print('len(categorical_headers): ' + str(len(new_headers)))
        print(new_headers)
        df_new = pd.DataFrame(index=df.index, columns=new_headers);
        df_new = df_new.fillna(0);
        for index, row in df.iterrows():
            val = str(row[featurename]);
            assert ( val in feature_values), "The current value is not in the list of possible value for this feature: %s" % str(val)
            col_new = featurename + '_' + str(val);
            df_new.at[index, col_new] = 1;
        return df_new;


    def __categorizeValues(self, df, featurename):
        group_values = self.options.getFeatureCategories(featurename);
        df_new = self.__categorizeMulti(df, featurename, group_values);
        # if len(group_values) == 2:
        #     df_new = self.__categorizeBinary(df, featurename, group_values)
        # elif len(group_values) > 2:
        #     df_new = self.__categorizeMulti(df, featurename, group_values);
        # else:
        #     print('there should not exist a feature with a number of categories smaller than 2...exit')
        #     sys.exit()
        return df_new;

    def __encodeCategoricalFeatures(self, df):
        categorical_features = self.options.getCategoricalFeatures();
        for feat in sorted(categorical_features):
            print('encode feature: ' + str(feat));
            df_new = self.__categorizeValues(df, feat);
            df = pd.concat([df, df_new], axis=1);
            df = df.drop(feat, axis=1);
        print('df: ' + str(df.shape))
        return df;


    def __encodeBinarizableFeature(self, df, name):
        print('binarize feature: ' + str(name));
        columns = list(df.columns);
        if name in columns:
            if name == 'Verweildauer':
                df[name] = df[name].apply(self.options.getLOSState);
                df_binary = self.__categorizeValues(df, name);
            elif name == 'Eintrittsalter':
                df[name] = df[name].apply(self.options.getAgeState);
                df_binary = self.__categorizeValues(df, name);
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
            if h == 'Hauptdiagnose' or h == 'main_diag':
                print('preprocess main diagnosis...')
                df[h] = df[h].apply(self.__prepareHauptdiagnose);
            elif h == 'AufnehmOE':
                df[h] = df[h].apply(self.__prepareOE);
            elif h == 'EntlassOE':
                df[h] = df[h].apply(self.__prepareOE);
            elif h == 'DRGCode':
                df[h] = df['DRGCode'].apply(self.__prepareMDC);
        return df;


    def encodeFeatures(self):
        encoding = self.options.getEncodingScheme();
        assert encoding is not None, 'an encoding algorithm has to be selected..exit';
        assert encoding in ['categorical', 'binary', 'embedding'], 'feature encoding scheme is not known...please select one of the following: categorical, binary, embedding';

        dir_data = self.options.getDirData();
        data_prefix = self.options.getDataPrefix();
        dataset = self.options.getDatasetName();
        print('encode features: ' + str(encoding))
        strFilename_in = dataset + '_REST_' + self.filename_options_in;
        strFilename_out = strFilename_in + '_' + encoding;
        filename_data_in = dir_data + 'data_' + data_prefix + '_' + strFilename_in + '.csv';
        filename_data_out = dir_data + 'data_' + data_prefix + '_' + strFilename_out + '.csv';

        df = pd.read_csv(filename_data_in);
        df = self.__preprocessFeatureEncoding(df);

        if encoding == 'categorical' or encoding == 'binary':
            df = self.__encodeCategoricalFeatures(df);
            if encoding == 'binary':
                binarizable_features = self.options.getCountFeaturesToBinarize();
                for bin_feat in binarizable_features:
                    df = self.__encodeBinarizableFeature(df, bin_feat);
        else:
            print('encoding scheme is not known... no encoding applied')

        print('encoded: df.shape: ' + str(df.shape))
        df.to_csv(filename_data_out, line_terminator='\n', index=False);


    def encodeFeaturesNZ(self):
        encoding = self.options.getEncodingScheme();
        assert encoding is not None, 'an encoding algorithm has to be selected..exit';
        assert encoding in ['categorical', 'binary', 'embedding'], 'feature encoding scheme is not known...please select one of the following: categorical, binary, embedding';

        dir_data = self.options.getDirData();
        data_prefix = self.options.getDataPrefix();
        dataset = self.options.getDatasetName();
        filename_data_in = dir_data + 'data_' + data_prefix + '_' + dataset + '_discharge.csv';
        filename_data_out = dir_data + 'data_' + data_prefix + '_' + dataset + '_discharge_' + encoding + '.csv';
        df = pd.read_csv(filename_data_in);
        df = self.__preprocessFeatureEncoding(df);

        if encoding == 'categorical' or encoding == 'binary':
            df = self.__encodeCategoricalFeatures(df);
            if encoding == 'binary':
                binarizable_features = self.options.getCountFeaturesToBinarize();
                for bin_feat in binarizable_features:
                    df = self.__encodeBinarizableFeature(df, bin_feat);
        else:
            print('encoding scheme is not known... no encoding applied')

        print('encoded: df.shape: ' + str(df.shape))
        df.to_csv(filename_data_out, line_terminator='\n', index=False);

