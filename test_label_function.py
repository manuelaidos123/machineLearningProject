import unittest
import numpy as np
from main import label
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class TestLabelFunction(unittest.TestCase):

    def test_label_encoding_and_stratification(self):
        # Load the actual dataset from './credit/crx.data'
        dataset = np.genfromtxt('./credit/crx.data', delimiter=',', dtype='str')
        descriptive = dataset[:, :-1]  # Features
        target = dataset[:, -1]  # Target variable

        # Define categorical columns based on your dataset
        categorical_cols = [0, 3, 4, 5, 6, 8, 9, 11]

        # Apply label encoding
        label_encoder = LabelEncoder()
        expected_descriptive_encoded = descriptive.copy()
        for col in categorical_cols:
            expected_descriptive_encoded[:, col] = label_encoder.fit_transform(descriptive[:, col])

        # Perform label encoding and splitting with stratification
        train_size = 0.2  # Replace with your desired test size
        descriptive_train, descriptive_test, target_train, target_test = train_test_split(
            descriptive, target, test_size=train_size, random_state=0, stratify=target
        )

        # Check if label encoding is applied correctly:
        # Instead of comparing encoded values directly, sort both arrays by the first categorical column

        first_cat_col = categorical_cols[0]
        descriptive_train_sorted = descriptive_train[descriptive_train[:, first_cat_col].argsort()]
        expected_descriptive_encoded_sorted = expected_descriptive_encoded[expected_descriptive_encoded[:, first_cat_col].argsort()]

        for col in categorical_cols:
            self.assertTrue(np.array_equal(descriptive_train_sorted[:, col], expected_descriptive_encoded_sorted[:, col]))

        # Check if label distribution is preserved even after splitting:
        target_distribution = np.bincount(target)
        train_target_distribution = np.bincount(target_train)
        test_target_distribution = np.bincount(target_test)

        self.assertTrue(np.allclose(train_target_distribution / target_distribution, test_target_distribution / target_distribution))

if __name__ == '__main__':
    unittest.main()

