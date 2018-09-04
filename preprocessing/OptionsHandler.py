



class PatrecDataOptionsHandler:

    def __init__(self, type):
        self.dataset_training = None;
        self.dataset_testing = None;
        self.data_preprocessing = None;
        self.filterKey_training = None;
        self.filterValue_training = None;
        self.filterKey_testing = None;
        self.filterValue_testing = None;
        self.extra_training = None;
        self.extra_testing = None;
        self.clf_name = None;
        self.clf_options = None;
        # 'extra_options': None
        return;

    def getFilenameStrDataset(self, strDataset, preprocessing, filterKey=None, filterValue=None, extraStr=None):
        strFilename = ''
        if filterKey is not None:
            strFilename = preprocessing + '_' + strDataset + '_' + str(filterKey) + '_' + str(filterValue);
        else:
            strFilename = preprocessing + '_' + strDataset;

        if extraStr is not None:
            strFilename = strFilename + '_' + extraStr;

        return strFilename;
