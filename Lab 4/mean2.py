import pandas as pd
import numpy as np
df = pd.DataFrame({ 'Age' : [25, 30, np.nan, 40, 35 ], 'Department' : ['HR', 'Finance', 'Finance', np.nan , 'IT'] })
print("Original Dataset (with missing values):")
print(df)
df_ffill = df.copy()
df_ffill.ffill(inplace=True)
print(df_ffill) 
