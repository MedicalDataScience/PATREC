
import pandas as pd

from utils.DatasetSplitter import DatasetSplitter
from utils.DatasetFilter import DatasetFilter


class Dataset:
    
    def __init__(self, dataset_options):
        self.options = dataset_options;
        self.df = None;
        self.columns = None;
        return;


    def _getDf(self):
        filename = self.options.getFilename()
        df = pd.read_csv(filename); 
        self.df = df;
    

    def _getColumns(self):
        cols = list(self.df.columns);
        self.columns = cols;


    def getFilename(self):
        return self.options.getFilename();


    def getColumns(self):
        if self.df is None:
            self._getDf();
        if self.columns is None:
            self._getColumns();
        return self.columns;    


    def getDf(self):
        if self.df is None:
            self._getDf();
        return self.df;

    # maybe stop creating separate files for filtered datasets and just create the df on the flyx
    def filterData(self, filterKey):
        filter = DatasetFilter(self.options)
        filter.filterDataBinaryColumns(filterKey)


    def splitDatasetIntoTrainingTestingSet(self):
        datasplitter = DatasetSplitter(self.options)
        datasplitter.splitDatasetIntoTrainingTesting();


