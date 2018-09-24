
import pandas as pd



class DatasetSplitter:

    def __init__(self, options_dataset):
        self.options = options_dataset;
        return;


    def __getFilenameOptionStr(self):
        dataset = self.options.getDatasetName();
        featureset = self.options.getFeatureSet();
        encoding = self.options.getEncodingScheme();
        grouping = self.options.getGroupingName();
        filename_options_in = featureset + '_' + encoding + '_' + grouping;

        if filename_options_in is None:
            print('filename options must not be None: ')
            print('filename_options_in: '  + str(filename_options_in))
        strFilename = dataset + '_' + filename_options_in;
        return strFilename


    def __splitDataset(self, df):
        df_pos = df.loc[df['Wiederkehrer'] == 1]
        df_neg = df.loc[df['Wiederkehrer'] == 0]
        df_pos = df_pos.sample(frac=1);
        df_neg = df_neg.sample(frac=1);
        return [df_pos, df_neg];


    def __splitDataTrainingTestingAll(self, df):
        ratio_training_samples = self.options.getRatioTrainingSamples();

        [df_pos, df_neg] = self.__splitDataset(df)
        num_pos_samples = df_pos.shape[0];
        num_pos_samples_training = int(round(ratio_training_samples * num_pos_samples));
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

    def __splitDataAll(self, df):
        dir_data = self.options.getDirData();
        filename_option_str = self.__getFilenameOptionStr()
        filename_data_out_pos = dir_data + 'data_' + filename_option_str + '_pos_all' + '.csv';
        filename_data_out_neg = dir_data + 'data_' + filename_option_str + '_neg_all' + '.csv';

        df_pos = df.loc[df['Wiederkehrer'] == 1]
        df_neg = df.loc[df['Wiederkehrer'] == 0]
        df_pos = df_pos.sample(frac=1);
        df_neg = df_neg.sample(frac=1);

        df_neg.to_csv(filename_data_out_neg, line_terminator='\n', index=False);
        df_pos.to_csv(filename_data_out_pos, line_terminator='\n', index=False);


    def __splitDataBalanced(self, df):
        dir_data = self.options.getDirData();
        filename_option_str = self.__getFilenameOptionStr()
        filename_data_out_pos = dir_data + 'data_' + filename_option_str + '_pos_balanced' + '.csv';
        filename_data_out_neg = dir_data + 'data_' + filename_option_str + '_neg_balanced' + '.csv';
        filename_data_out = dir_data + 'data_' + filename_option_str + '_balanced' + '.csv';

        df_pos = df.loc[df['Wiederkehrer'] == 1]
        df_neg = df.loc[df['Wiederkehrer'] == 0]
        df_pos = df_pos.sample(frac=1);
        df_neg = df_neg.sample(frac=1);

        num_pos_samples = df_pos.shape[0];
        df_neg = df_neg.iloc[:num_pos_samples];

        print('df_pos:' + str(df_pos.shape))
        print('df_neg: ' + str(df_neg.shape));

        df_neg.to_csv(filename_data_out_neg, line_terminator='\n', index=False);
        df_pos.to_csv(filename_data_out_pos, line_terminator='\n', index=False);

        df_balanced = df_pos.append(df_neg);
        df_balanced = df_balanced.sample(frac=1);
        df_balanced.to_csv(filename_data_out, line_terminator='\n', index=False);


    def __splitDatasetIntoTrainingTestingAll(self, df):
        dir_data = self.options.getDirData();
        filename_option_str = self.__getFilenameOptionStr()
        [training, testing] = self.__splitDataTrainingTestingAll(df);
        filename_data_out_pos_training = dir_data + 'data_' + filename_option_str + '_pos_training' + '.csv';
        filename_data_out_neg_training = dir_data + 'data_' + filename_option_str + '_neg_training' + '.csv';
        filename_data_out_pos_testing = dir_data + 'data_' + filename_option_str + '_pos_testing' + '.csv';
        filename_data_out_neg_testing = dir_data + 'data_' + filename_option_str + '_neg_testing' + '.csv';
        training[1].to_csv(filename_data_out_neg_training, line_terminator='\n', index=False);
        training[0].to_csv(filename_data_out_pos_training, line_terminator='\n', index=False);
        testing[1].to_csv(filename_data_out_neg_testing, line_terminator='\n', index=False);
        testing[0].to_csv(filename_data_out_pos_testing, line_terminator='\n', index=False);


    def splitDatasetIntoTrainingTesting(self):
        dir_data = self.options.getDirData();
        filename_option_str = self.__getFilenameOptionStr()
        filename_data_in = dir_data + 'data_' + filename_option_str + '.csv';
        df = pd.read_csv(filename_data_in);

        self.__splitDatasetIntoTrainingTestingAll(df);
        self.__splitDataAll(df);
        self.__splitDataBalanced(df);