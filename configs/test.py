import pandas as pd

df = pd.read_csv("submissions\EffUNet_submission.csv")
print(df.describe())