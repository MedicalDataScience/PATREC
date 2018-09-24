import sys

class DatasetOptions:

    def __init__(self, options):
        self.filename_data_1 = '/scicore/home/vogtju/GROUP/PATREC_USB/PR15_2012-15___Anonym18Konsent.csv';
        self.filename_data_2 = '/scicore/home/vogtju/GROUP/PATREC_USB/PR15_2016-17___Anonym18Konsent.csv';

        self.dir_data = options['dir_data'];
        self.dataset = options['dataset'];
        self.subgroups = options['subgroups'];
        self.featureset = options['featureset'];
        self.options_featureset = options['options_featureset'];
        self.grouping = options['grouping'];
        self.options_grouping = options['options_grouping'];
        self.encoding = options['encoding'];
        self.options_encoding = options['options_encoding'];
        self.options_filtering = options['options_filtering']
        self.chunksize = options['chunksize'];
        self.ratio_training_samples = options['ratio_training_samples'];
        self.filename = self._getFilename();
        return;


    def _getFilename(self):
        str = self.dir_data + '/data_' + self.dataset;
        
        str = str + '_' + self.featureset;
        #if self.options_featureset is not None:
        #    str = str + self.options_featureset;
        
        str = str + '_' + self.encoding;
        if self.options_encoding is not None:
            str = str + self.options_grouping;

        str = str + '_' + self.grouping;
        if self.options_grouping is not None:
            str = str + self.options_grouping;

        if self.options_filtering is not None:
            str = str + '_' + self.options_filtering;

        str = str + '.csv';
        return str;


    def getDirData(self):
        return self.dir_data;


    def getDatasetName(self):
        return self.dataset;


    def getSubgroups(self):
        return self.subgroups;


    def getFeatureSet(self):
        return self.featureset;


    def getFeatureSetOptions(self):
        return self.options_featureset;


    def getEncodingScheme(self):
        return self.encoding;


    def getChunkSize(self):
        return self.chunksize;


    def getRatioTrainingSamples(self):
        return self.ratio_training_samples;


    def getFilenameRawData(self):
        if self.dataset == '20122015':
            filename = self.filename_data_1;
        elif self.dataset == '20122015':
            filename = self.filename_data_2;
        else:
            print('dataset is not known...exit')
            sys.exit()
        return filename;


    def getFilename(self):
        if self.filename is None:
            self.filename = self._getFilename();
        return self.filename;
            
