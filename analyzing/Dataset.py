
import pandas as pd

class Dataset:
    
    def __init__(self, dataset_options):
        self.options = dataset_options;
        self.data = None;
        self.columns = None;
        return;


    def _getData(self):
        filename = self.options.getFilename()
        df = pd.read_csv(filename); 
        self.data = df;
    

    def _getColumns(self):
        cols = list(self.data.columns);
        self.columns = cols;


    def getFilename(self):
        return self.options.getFilename();


    def getColumns(self):
        if self.columns is None:
            self._getColumns();
        return self.columns;    


    def getData(self):
        if self.data is None:
            self._getData();
        return self.data;    

