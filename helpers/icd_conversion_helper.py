import os
import inspect
import pandas as pd


class ICDConversionHelper:
    ICD9_TO_ICD10_CONVERSION_MAPPING = 'icd9_to_icd10.csv'
    ICD10_6_TO_ICD10_8_CONVERSION_MAPPING = 'icd10_6_to_icd10_8.csv'

    def __init__(self):
        curr_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        self.ICD9_to_ICD10 = pd.read_csv(curr_dir + "/" + self.ICD9_TO_ICD10_CONVERSION_MAPPING)
        self.ICD10_6_to_ICD10_8 = pd.read_csv(curr_dir + "/" + self.ICD10_6_TO_ICD10_8_CONVERSION_MAPPING)

    def convert_icd9_to_icd10(self, icd9_code, diag_typ):
            df = self.ICD9_to_ICD10.query("ICD9 == '{}'".format(icd9_code))

            if df.shape[0] == 1:
                return df["ICD10"].values[0]
            elif df.shape[0] == 0:
                raise KeyError("ICD10 code not found for key '{}'!".format(icd9_code))
            elif df.shape[0] > 1:
                # diag_typ is O for operation, assume a procedure code
                if diag_typ == "O":
                    for ind, row in df.iterrows():
                        if len(row.ICD10) == 7:
                            return row["ICD10"]
                # diag_typ is not O, use diagnosis code
                else:
                    for ind, row in df.iterrows():
                        if len(row.ICD10) < 7:
                            return row["ICD10"]

                # No suitable code found
                raise KeyError("No suitable code found for one to many mapping for key '{}'".format(icd9_code))

    def convert_icd10_6_to_icd10_8(self, icd10code):
        df = self.ICD10_6_to_ICD10_8.query("ICD10_6 == '{}'".format(icd10code))

        # If no rows are found, there is nothing to do so just return the same code
        if df.shape[0] == 0:
            return icd10code
        else:
            return df["ICD10_8"].values[0]

    def get_icd9_codes(self):
        return self.ICD9_to_ICD10['ICD9'].values

    def get_icd10_8_codes(self):
        return self.ICD9_to_ICD10['ICD10'].values + self.ICD10_6_to_ICD10_8['ICD10_8'].values