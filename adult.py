import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def clean_data_adult():
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                    'hours_per_week', 'native_country', 'income']

    adult_file = 'adult/adult.data' 
    df = pd.read_csv(adult_file, header=None, names=column_names)
    df.replace('?', np.nan, inplace=True)

    numeric_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].mean(), inplace=True)

    categorical_cols = ['workclass', 'education', 'marital_status', 'occupation', 
                        'relationship', 'race', 'sex', 'native_country', 'income']
    df.dropna(subset=categorical_cols, inplace=True)

    return df

def label_adult(desc, targ, percentage, cols):
    descriptive, target = desc.copy(), targ.copy()
    labelEncoder = LabelEncoder()
    for i in cols:
        descriptive[:, i] = labelEncoder.fit_transform(descriptive[:, i])
    descriptive_train, descriptive_test, target_train, target_test = train_test_split(
        descriptive, target, test_size=percentage, random_state=0)
    return descriptive_train, descriptive_test, target_train, target_test

def standard_adult(desc, targ, percentage, cols):
    descriptive, target = desc.copy(), targ.copy()
    labelEncoder = LabelEncoder()
    for i in cols:
        descriptive[:, i] = labelEncoder.fit_transform(descriptive[:, i])
    descriptive_train, descriptive_test, target_train, target_test = train_test_split(
        descriptive, target, test_size=percentage, random_state=0)
    standard_scaler = StandardScaler()
    descriptive_train[:, :] = standard_scaler.fit_transform(descriptive_train[:, :])
    descriptive_test[:, :] = standard_scaler.fit_transform(descriptive_test[:, :])
    return descriptive_train, descriptive_test, target_train, target_test

def one_hot_adult(desc, targ, percentage, cols):
    descriptive, target = desc.copy(), targ.copy()
    column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cols)], remainder='passthrough')
    descriptive = np.array(column_transformer.fit_transform(descriptive).toarray())
    descriptive_train, descriptive_test, target_train, target_test = train_test_split(
        descriptive, target, test_size=percentage, random_state=0)
    return descriptive_train, descriptive_test, target_train, target_test

def one_hot_standard_adult(desc, targ, percentage, cols):
    descriptive, target = desc.copy(), targ.copy()
    column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cols)], remainder='passthrough')
    descriptive = np.array(column_transformer.fit_transform(descriptive).toarray())
    descriptive_train, descriptive_test, target_train, target_test = train_test_split(
        descriptive, target, test_size=percentage, random_state=0)
    standard_scaler = StandardScaler()
    descriptive_train = standard_scaler.fit_transform(descriptive_train)
    descriptive_test = standard_scaler.transform(descriptive_test)

    return descriptive_train, descriptive_test, target_train, target_test
def naive_bayes_adult(descriptive_train, descriptive_test, target_train, target_test):
    classifier = GaussianNB()
    classifier.fit(descriptive_train, target_train)
    prediction = classifier.predict(descriptive_test)
    accuracy = accuracy_score(target_test, prediction)
    matrix = confusion_matrix(target_test, prediction)
    return accuracy, matrix

def knn_adult(descriptive_train, descriptive_test, target_train, target_test):
    classifier = KNeighborsClassifier(algorithm='ball_tree')
    classifier.fit(descriptive_train, target_train)
    prediction = classifier.predict(descriptive_test)
    accuracy = accuracy_score(target_test, prediction)
    matrix = confusion_matrix(target_test, prediction)
    return accuracy, matrix

def decision_tree_adult(descriptive_train, descriptive_test, target_train, target_test):
    classifier = DecisionTreeClassifier()
    classifier.fit(descriptive_train, target_train)
    prediction = classifier.predict(descriptive_test)
    accuracy = accuracy_score(target_test, prediction)
    matrix = confusion_matrix(target_test, prediction)
    return accuracy, matrix

def random_forest_adult(descriptive_train, descriptive_test, target_train, target_test):
    classifier = RandomForestClassifier()
    classifier.fit(descriptive_train, target_train)
    prediction = classifier.predict(descriptive_test)
    accuracy = accuracy_score(target_test, prediction)
    matrix = confusion_matrix(target_test, prediction)
    return accuracy, matrix

if __name__ == "__main__":
    df_adult = clean_data_adult()

    descriptive = df_adult.iloc[:, :-1].values
    target = df_adult.iloc[:, -1].values

    encoder = LabelEncoder()
    target = encoder.fit_transform(target)

    categorical_cols_indices = [i for i, col in enumerate(df_adult.columns[:-1]) if df_adult[col].dtype == 'object']

percentages = [0.15, 0.30, 0.5]

if __name__ == "__main__":
    df_adult = clean_data_adult()

    descriptive = df_adult.iloc[:, :-1].values
    target = df_adult['income'].values

    columns = []
    for i, cols in enumerate(df_adult.columns[:-1]):
        if df_adult[cols].dtypes == 'object': columns.append(i)

    encoder = LabelEncoder()
    target = encoder.fit_transform(target)

    models = {
        "Naive Bayes": naive_bayes_adult,
        " KNeighbors ": knn_adult,
        " Decision Tree ": decision_tree_adult,
        " Random Forest ": random_forest_adult,
    }

    precision_methods = {
        "labelencoder": label_adult,
        "labelEncoder + standardScaler": standard_adult,
        "labelEncoder + one-hot": one_hot_adult,
        "labelencoder + one-hot + standard": one_hot_standard_adult
    }

    percentagens = [0.15, 0.30, 0.5]

    for name_model, func_model in models.items():
        print(f"{name_model:=^60}")
        for name_precision, func_precision in precision_methods.items():
            print(f"{name_precision:.^50}")
            for percentagem in percentagens:
                descriptive_train, descriptive_test, target_train, target_test = func_precision(descriptive, target, percentagem, columns)
                accuracy, matrix = func_model(descriptive_train, descriptive_test, target_train, target_test)
                print(f"[{percentagem:.2%}]: accuracy: {accuracy:.4f}")
                print("Confusion Matrix:")
                print(matrix)
                print()
    print("Finished.")


    