import pandas as pd
import numpy as np



df = pd.read_csv("Escherichia_isolates.csv")
dum_df = pd.get_dummies(df, columns=["antibiotic"], prefix=["res"])
dum_df.head()