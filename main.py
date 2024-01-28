#! /bin/env python


import pandas as pd 

cols = [
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
            "A9",
            "A10",
            "A11",
            "A12",
            "A13",
            "A14",
            "A15",
            "A16", ]

credit_file = "./credit/crx.data"
df = pd.read_csv(credit_file,names=cols, header=0)

print(df.head())
print(df['A1'].unique())
