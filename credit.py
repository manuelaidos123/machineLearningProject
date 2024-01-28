import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np

def clean_data_credit():
    cols = ["A1", "A2", "A3", "A4", "A5", "A6",
            "A7", "A8", "A9", "A10", "A11", "A12",
            "A13", "A14", "A15", "A16", ]
    order_col = [ "A2", "A3", "A8", "A11", "A14", "A15",
"A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13", "A16", ]
    credit_file = "./credit/crx.data"
    df = pd.read_csv(credit_file,names=cols, header=0)
    df.replace('?', np.nan, inplace=True)
    df['A2'][df['A2'].isna() == True]
    df['A2'] = pd.to_numeric(df['A2'], errors='coerce')
    df['A2'] = df['A2'].fillna(df.A2.mean())
    df['A14'] = pd.to_numeric(df['A2'], errors='coerce')
    df['A14'] = df['A14'].fillna(df.A2.mean())
    cols_with_missing_values = [1,4,5,6,7,9,10,12,13]
    df.dropna(subset=[f'A{col}' for col in cols_with_missing_values], inplace=True)

    return df[order_col]

def label(desc, targ, percentagem, cols):
    descriptive, target = desc.copy(), targ.copy()

    labelEncoder = LabelEncoder()
    for i in cols:
        descriptive[:,i] = labelEncoder.fit_transform(descriptive[:,i])

    descriptive_train, descriptive_test, target_train, target_test = train_test_split(descriptive, target, test_size=percentagem, random_state=0)
    return descriptive_train, descriptive_test, target_train, target_test

def standard(desc, targ, percentagem, cols):
    descriptive, target = desc.copy(), targ.copy()
    labelEncoder = LabelEncoder()
    for i in cols:
        descriptive[:,i] = labelEncoder.fit_transform(descriptive[:,i])

    descriptive_train, descriptive_test, target_train, target_test = train_test_split(descriptive, target, test_size=percentagem, random_state=0)

    standard_scaler = StandardScaler()
    descriptive_train[:,:] = standard_scaler.fit_transform(descriptive_train[:,:])
    descriptive_test[:,:] = standard_scaler.fit_transform(descriptive_test[:,:])

    return descriptive_train, descriptive_test, target_train, target_test

def one_hot(desc, targ, percentagem, cols):
    descriptive, target = desc.copy(), targ.copy()

    column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cols)],
                                           remainder = 'passthrough'
                                           )

    descriptive = np.array(column_transformer.fit_transform(descriptive))
    descriptive_train, descriptive_test, target_train, target_test = train_test_split(descriptive, target, test_size=percentagem, random_state=0)

    return descriptive_train, descriptive_test, target_train, target_test

def one_hot_standard(desc, targ, percentagem, cols):
    descriptive, target = desc.copy(), targ.copy()

    column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cols)],
                                           remainder = 'passthrough'
                                           )
    descriptive = np.array(column_transformer.fit_transform(descriptive))
    descriptive_train, descriptive_test, target_train, target_test = train_test_split(descriptive, target, test_size=percentagem, random_state=0)

    standard_scaler = StandardScaler()
    descriptive_train[:,:6] = standard_scaler.fit_transform(descriptive_train[:,:6])
    descriptive_test[:,:6] = standard_scaler.fit_transform(descriptive_test[:,:6])

    return descriptive_train, descriptive_test, target_train, target_test

def naive_bayes(descriptive_train, descriptive_test, target_train, target_test):
    classifier = GaussianNB()
    classifier.fit(descriptive_train, target_train)
    prediction = classifier.predict(descriptive_test)
    accuracy = accuracy_score(target_test, prediction)
    matrix = confusion_matrix(target_test, prediction)
    return accuracy, matrix

def knn(descriptive_train, descriptive_test, target_train, target_test):
    classifier = KNeighborsClassifier()
    classifier.fit(descriptive_train, target_train)
    prediction = classifier.predict(descriptive_test)
    accuracy = accuracy_score(target_test, prediction)
    matrix = confusion_matrix(target_test, prediction)
    return accuracy, matrix

def decision_tree(descriptive_train, descriptive_test, target_train, target_test):
    classifier = DecisionTreeClassifier()
    classifier.fit(descriptive_train, target_train)
    prediction = classifier.predict(descriptive_test)
    accuracy = accuracy_score(target_test, prediction)
    matrix = confusion_matrix(target_test, prediction)
    return accuracy, matrix

def random_forest(descriptive_train, descriptive_test, target_train, target_test):
    classifier = RandomForestClassifier()
    classifier.fit(descriptive_train, target_train)
    prediction = classifier.predict(descriptive_test)
    accuracy = accuracy_score(target_test, prediction)
    matrix = confusion_matrix(target_test, prediction)
    return accuracy, matrix


if __name__ == "__main__":
    df = clean_data_credit()


    descriptive = df.iloc[:,0:15].values
    target = df.iloc[:,15].values

    columns = []
    for i, cols in enumerate(df.columns[:-1]):
        if df[cols].dtypes == 'object': columns.append(i)

    encoder = LabelEncoder()
    target = encoder.fit_transform(target)

    models = {
            " Naive Bayes ": naive_bayes,
            " KNeighbors ": knn,
            " Decision Tree ": decision_tree,
            " Random Forest ": random_forest,
            }

    precision = {
            "labelencoder": label,
            "labelEncoder + standardScaler": standard,
            "labelEncoder + one-hot": one_hot,
            "labelencoder + one-hot + standard": one_hot_standard
            }

    percentagens = [0.15, 0.30, 0.5]

    for name_model, func_model in models.items():
        print(f"{name_model:=^60}")
        for name_precision, func_precision in precision.items():
            print(f"{name_precision:.^50}")
            for percentagem in percentagens:
                descriptive_train, descriptive_test, target_train, target_test = func_precision(descriptive, target, percentagem, columns)
                accuracy, matrix = func_model(descriptive_train, descriptive_test, target_train, target_test)
                print(f"[{percentagem:.2%}]: accuracy: {accuracy:.4f}")
                print("Confusion Matrix:")
                print(matrix)
                print()
    print("Finished.")

