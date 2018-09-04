import sys
import pandas as pd



class FeatureCreator:

    def __init__(self, dir_data, dataset, filename_options_in, filename_options_out, subgroups, names_additional_features=None, chunksize=10000):
        self.dir_data = dir_data;
        self.dataset = dataset;
        self.subgroups = subgroups;
        self.filename_options_in = filename_options_in;
        self.filename_options_out = filename_options_out;
        self.additional_features = names_additional_features;
        self.chunksize = chunksize;
        return;


    def __getFilenameOptionStr(self):
        if self.filename_options_in is None or self.filename_options_out is None:
            print('filename options must not be None: ')
            print('filename_options_in: '  + str(self.filename_options_in))
            print('filename_options_out: ' + str(self.filename_options_out))
        strFilenameIn = self.dataset + '_REST_' + self.filename_options_in;
        strFilenameOut = self.dataset + '_REST_' + self.filename_options_out;
        return [strFilenameIn, strFilenameOut]


    def __calculatePreviousVisits(self, df):
        df_sorted = df.sort_values(by=['Patient', 'Aufnahmedatum']);
        patient_ids = df_sorted['Patient'].unique();
        df['previous_visits'] = 0;
        for p_id in patient_ids:
            df_patient = df_sorted.loc[df_sorted['Patient'] == p_id];
            cnt_visits = 0;
            for k,row in df_patient.iterrows():
                df.at[k,'previous_visits'] = cnt_visits;
                cnt_visits += 1;
        return df['previous_visits'];

    def __generateAdditionalFeatures(self, df):
        for newfeature in self.additional_features:
            print(newfeature)
            if newfeature == 'ratio_los_age':
                df['ratio_los_age'] = df['Verweildauer'] / df['Eintrittsalter'];
            elif newfeature == 'ratio_numDK_age':
                df['ratio_numDK_age'] = df['numDK'] / df['Eintrittsalter'];
            elif newfeature == 'ratio_los_numDK':
                df['ratio_los_numDK'] = df['Verweildauer'] / df['numDK'];
            elif newfeature == 'ratio_numCHOP_age':
                df['ratio_numCHOP_age'] = df['numCHOP'] / df['Eintrittsalter'];
            elif newfeature == 'ratio_los_numOE':
                df['ratio_los_numOE'] = df['Verweildauer'] / df['numOE'];
            elif newfeature == 'ratio_numOE_age':
                df['ratio_numOE_age'] = df['numOE'] / df['Eintrittsalter'];
            elif newfeature == 'mult_los_numCHOP':
                df['mult_los_numCHOP'] = df['Verweildauer'] * df['numCHOP'];
            elif newfeature == 'mult_equalOE_numDK':
                df['mult_equalOE_numDK'] = df['equalOE'] * df['numDK'];
            elif newfeature == 'previous_visits':
                df[newfeature] = self.__calculatePreviousVisits(df);
            else:
                print('this additional feature is not yet implemented...sorry')
                sys.exit()
        return df;


    def __createFeatureNumOccurrences(self, name_subgroup):
        df_num = pd.DataFrame(columns=['num' + name_subgroup]);
        strFilenameIn = self.dataset + '_' + name_subgroup + '_clean';
        filename_data_subgroup_in = self.dir_data + 'data_' + strFilenameIn + '.csv';
        data_reader = pd.read_csv(filename_data_subgroup_in, chunksize=self.chunksize);
        for k, chunk in enumerate(data_reader):
            chunk = chunk.drop('Fall', axis=1);
            chunk_new = pd.DataFrame(index=chunk.index, columns=['num'+name_subgroup]);
            chunk_new['num'+name_subgroup] = chunk.sum(axis=1)
            df_num = df_num.append(chunk_new)
        print('df_num: ' + str(df_num.shape))
        return df_num;


    def __addFeaturesFromCleanData(self, df):
        #create cnt features: numCHOP, numDK, numOE
        for sub in self.subgroups:
            feature_name = 'num' + sub;
            df_newfeature = self.__createFeatureNumOccurrences(sub);
            df[feature_name] = df_newfeature.values;
            print('df.shape: ' + str(df.shape))
            if sub == 'OE':
                df_newfeature = pd.DataFrame(index=df.index, columns=['equalOE']);
                df_newfeature = df_newfeature.fillna(0);
                df_newfeature['equalOE'] = (df['EntlassOE'] == df['AufnehmOE']).astype(int);
                df['equalOE'] = df_newfeature.values;
        print('df.shape: ' + str(df.shape))
        return df;


    def addFeatures(self):
        [filename_str_in, filename_str_out] = self.__getFilenameOptionStr()
        filename_data_in = self.dir_data + 'data_' + filename_str_in + '.csv';
        filename_data_out = self.dir_data + 'data_' + filename_str_out + '.csv';
        df = pd.read_csv(filename_data_in);

        df_newfeatures = self.__addFeaturesFromCleanData(df);
        if self.additional_features is not None:
            df_additional_features = self.__generateAdditionalFeatures(df_newfeatures);
            df_newfeatures = df_additional_features;

        df_newfeatures.to_csv(filename_data_out, line_terminator='\n', index=False);