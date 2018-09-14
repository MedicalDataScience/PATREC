

class DatasetOptions:

    def __init__(self, options):
        self.dir_data = options['dir_data'];
        self.dataset = options['dataset'];
        self.featureset = options['featureset'];
        self.options_featureset = options['options_featureset'];
        self.grouping = options['grouping'];
        self.options_grouping = options['options_grouping'];
        self.encoding = options['encoding'];
        self.options_encoding = options['options_encoding'];
        self.options_filtering = options['filtering']

        self.filename = self._getFilename();
        return;


    def _getFilename(self):
        str = self.dir_data + '/data_' + self.dataset;
        
        str = str + '_' + self.featureset;
        if self.options_featureset is not None:
            str = str + self.options_featureset;
        
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


    def getSrcDataset(self):
        return self.dataset;


    def getFilename(self):
        if self.filename is None:
            self.filename = self._getFilename();
        return self.filename;
            
