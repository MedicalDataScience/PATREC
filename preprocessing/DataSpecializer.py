
import pandas as pd



class DataSpecializer:

    def __init__(self, dir_data, dataset, filename_options_in, filename_options_out, names_additional_features=None, ratio_training_samples=0.85):
        self.dir_data = dir_data;
        self.dataset = dataset;
        self.filename_options_in = filename_options_in;
        self.filename_options_out = filename_options_out;
        self.additional_featues = names_additional_features;
        self.ratio_training_samples = ratio_training_samples;
        return;


    def __getFilenameOptionStr(self):
        if self.filename_options_in is None or self.filename_options_out is None:
            print('filename options must not be None: ')
            print('filename_options_in: '  + str(self.filename_options_in))
            print('filename_options_out: ' + str(self.filename_options_out))
        strFilenameIn = self.dataset + '_' + self.filename_options_in;
        strFilenameOut = self.dataset + '_' + self.filename_options_out;
        return [strFilenameIn, strFilenameOut]


    def __generateAdditionalFeatures(self, df):
        for newfeature in self.additional_features:
            if newfeature == 'ratio_los_age':
                df['ratio_los_age'] = df['Verweildauer'] / df['Eintrittsalter'];
            elif newfeature == 'ratio_numDK_age':
                df['ratio_numDK_age'] = df['numDK'] / df['Eintrittsalter'];
            elif newfeature == 'ratio_los_numDK':
                df['ratio_los_numDK'] = df['Verweildauer'] / df['numDK'];
            elif newfeature == 'ratio_numCHOP_age':
                df['ratio_numCHOP_age'] = df['numCHOP'] / df['Eintrittsalter'];
            elif newfeature == 'ratio_numCHOP_age':
                df['ratio_numCHOP_numDK'] = df['numCHOP'] / df['numDK'];
            elif newfeature == 'ratio_los_numOE':
                df['ratio_los_numOE'] = df['Verweildauer'] / df['numOE'];
            elif newfeature == 'ratio_numOE_age':
                df['ratio_numOE_age'] = df['numOE'] / df['Eintrittsalter'];
            elif newfeature == 'mult_los_numCHOP':
                df['mult_los_numCHOP'] = df['Verweildauer'] * df['numCHOP'];
            elif newfeature == 'mult_equalOE_numDK':
                df['mult_equalOE_numDK'] = df['equalOE'] * df['numDK'];
            else:
                print('this additional feature is not yet implemented...sorry')
                sys.exit()
        return df;


    def __splitDataTrainingTestingAll(self, df):
        [df_pos, df_neg] = self.__splitDataset(df)
        num_pos_samples = df_pos.shape[0];
        num_pos_samples_training = int(round(self.ratio_training_samples * num_pos_samples));
        num_pos_samples_testing = num_pos_samples - num_pos_samples_training;

        df_pos_training = df_pos.iloc[:num_pos_samples_training, :];
        df_pos_testing = df_pos.iloc[num_pos_samples_training:, :];
        print('df_pos_training: ' + str(df_pos_training.shape))
        print('df_pos_testing: ' + str(df_pos_testing.shape))

        df_neg_testing = df_neg.iloc[:num_pos_samples_testing, :];
        df_neg_training = df_neg.iloc[num_pos_samples_testing:, :];
        print('df_neg_training: ' + str(df_neg_training.shape))
        print('df_neg_testing: ' + str(df_neg_testing.shape))

        training = [df_pos_training, df_neg_training];
        testing = [df_pos_testing, df_neg_testing];
        return [training, testing]


    def __splitDatasetIntoTrainingTesting(self):
        [filename_in_str, filename_out_str] = self.__getFilenameOptionStr()
        filename_data_in = self.dir_data + 'data_' + filename_in_str + '.csv';
        df = pd.read_csv(filename_data_in);

        [training, testing] = self.__splitDataTrainingTestingAll(df);
        filename_data_out_pos_training = self.dir_data + 'data_' + filename_out_str + '_pos_training' + '.csv';
        filename_data_out_neg_training = self.dir_data + 'data_' + filename_out_str + '_neg_training' + '.csv';
        filename_data_out_pos_testing = self.dir_data + 'data_' + filename_out_str + '_pos_testing' + '.csv';
        filename_data_out_neg_testing = self.dir_data + 'data_' + filename_out_str + '_neg_testing' + '.csv';
        training[1].to_csv(filename_data_out_neg_training, line_terminator='\n', index=False);
        training[0].to_csv(filename_data_out_pos_training, line_terminator='\n', index=False);
        testing[1].to_csv(filename_data_out_neg_testing, line_terminator='\n', index=False);
        testing[0].to_csv(filename_data_out_pos_testing, line_terminator='\n', index=False);


    def addHandcraftedFeatures(self):
        [filename_str_in, filename_str_out] = self.__getFilenameOptionStr()
        filename_data_in = self.dir_data + 'data_' + filename_str_in + '.csv';
        filename_data_out = self.dir_data + 'data_' + filename_str_out + '.csv';
        df = pd.read_csv(filename_data_in);
        if self.additional_features is not None:
            df_additional_features = self.__generateAdditionalFeatures(df);
            df_additional_features.to_csv(filename_data_out, line_terminator='\n', index=False);


    def splitDatasetIntoTrainingTesting(self):
        self.__splitDatasetIntoTrainingTesting()