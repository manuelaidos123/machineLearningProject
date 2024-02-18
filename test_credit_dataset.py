import unittest
import pandas as pd
from main import credit_dataset

class TestCreditDataset(unittest.TestCase):

    def setUp(self):
        self.df = credit_dataset()

    def test_read_dataset_file(self):
        # Check if DataFrame is not empty
        self.assertFalse(self.df.empty, "DataFrame is empty")

    def test_missing_values_handling(self):
        # Check if there are any missing values in the DataFrame
        self.assertFalse(self.df.isnull().values.any(), "Missing values exist in the DataFrame")

    def test_correct_columns_and_data_types(self):
        # Define expected column names and data types
        expected_columns = ["A2", "A3", "A8", "A11", "A14", "A15", "A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13", "A16"]
        expected_data_types = {'A2': 'float64', 'A3': 'object', 'A8': 'object', 'A11': 'object', 'A14': 'object', 'A15': 'object',
                               'A1': 'object', 'A4': 'object', 'A5': 'object', 'A6': 'object', 'A7': 'object', 'A9': 'object',
                               'A10': 'object', 'A12': 'object', 'A13': 'object', 'A16': 'object'}

        # Check if columns match and data types are correct
        self.assertListEqual(list(self.df.columns), expected_columns, "Incorrect columns")
        self.assertDictEqual(self.df.dtypes.apply(lambda x: x.name).to_dict(), expected_data_types, "Incorrect data types")

if __name__ == '__main__':
    unittest.main()
