
import sys
from utils.Dataset import Dataset


class NeuralNetDatasetMaker:

    def __init__(self, mode, dataset_options, balanced_datasets=True):
        self.mode = mode;
        self.dataset_options = dataset_options;
        self.dataset = Dataset(self.dataset_options);
        self.balanced_datasets = balanced_datasets;
        return;



    def createDatasets(self):
        print('_getFilenameDatasetBalanced: ' + str(self.mode))
        filename_dataset_base = self.dataset_options.getFilename();

        if self.mode == 'traineval':
            if self.balanced_datasets:
                [df_training, df_testing] = self.dataset.getBalancedSubsetTrainingAndTesting();
                self.num_samples_train = df_training.shape[0];
                self.num_samples_validation = df_testing.shape[0];
                filename_train = filename_dataset_base[:-4] + '_balanced_train.csv'
                filename_eval = filename_dataset_base[:-4] + '_balanced_eval.csv'
                df_training.to_csv(filename_train, line_terminator='\n', index=False);
                df_testing.to_csv(filename_eval, line_terminator='\n', index=False);
                print(filename_train)
                print(filename_eval)
            else:
                [training, testing] = self.dataset.getTrainingAndTestingSet();
                df_training_pos = training[0];
                df_training_neg = training[1];
                df_eval_pos = testing[0];
                df_eval_neg = testing[1];
                self.num_samples_train = 2*int(df_training_neg.shape[0]);
                self.num_samples_validation = 2*int(df_eval_neg.shape[0]);
                filename_train_pos = filename_dataset_base[:-4] + '_train_pos.csv'
                filename_train_neg = filename_dataset_base[:-4] + '_train_neg.csv'
                filename_eval_pos = filename_dataset_base[:-4] + '_eval_pos.csv'
                filename_eval_neg = filename_dataset_base[:-4] + '_eval_neg.csv'
                df_training_pos.to_csv(filename_train_pos, line_terminator='\n', index=False);
                df_training_neg.to_csv(filename_train_neg, line_terminator='\n', index=False);
                df_eval_pos.to_csv(filename_eval_pos, line_terminator='\n', index=False);
                df_eval_neg.to_csv(filename_eval_neg, line_terminator='\n', index=False);
        else:
            if self.balanced_datasets:
                df_balanced = self.dataset.getBalancedSubSet();
                filename_dataset = filename_dataset_base[:-4] + '_balanced_' + self.mode + '.csv'
                df_balanced.to_csv(filename_dataset, line_terminator='\n', index=False);
                print(filename_dataset);
            else:
                print('no valid configuration of datasets and mode..exit')
                sys.exit();


    def _dfToFile(self, df, filename):
        list_df = [df[i:i + 10000] for i in range(0, df.shape[0], 10000)]
        list_df[0].to_csv(filename, index=False, line_terminator='\n')
        for l in list_df[1:]:
            l.to_csv(filename, index=False, line_terminator='\n', header=False, mode='a')

    def createDatasetsAutoEncoder(self):
        print('_getFilenameDatasetBalanced: ' + str(self.mode))
        filename_dataset_base = self.dataset_options.getFilename();

        df = self.dataset.getDf();
        df = df.sample(frac=1);
        print('num samples: ' + str(df.shape[0]))
        print('df.shape: ' + str(df.shape))
        num_samples = df.shape[0];
        ratio_train_test = self.dataset_options.getRatioTrainingSamples();
        df_train = df[:int(round(ratio_train_test*num_samples))];
        df_eval = df[int(round(ratio_train_test*num_samples)):];
        filename_train = filename_dataset_base[:-4] + '_balanced_train.csv'
        filename_eval = filename_dataset_base[:-4] + '_balanced_eval.csv'
        self._dfToFile(df_train, filename_train);
        self._dfToFile(df_eval, filename_eval);
        # df_train.to_csv(filename_train, line_terminator='\n', index=False);
        # df_eval.to_csv(filename_eval, line_terminator='\n', index=False);
