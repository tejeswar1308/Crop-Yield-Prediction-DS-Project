import pandas as pd

df = pd.DataFrame({ 'ID' : [1, 2, 2, 3, 4, 4], 'Name' : ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David'], 'Age' : [25, 30, 30, 35, 40, 40] })
print("Original Data:\n", df)

df_subset_id = df.drop_duplicates(subset=['ID'])
print("\n After Subset-Based Removal (ID):\n", df_subset_id)

df_subset_name = df.drop_duplicates(subset=['Name'])
print("\n After Subset-Based Removal (Name):\n", df_subset_name)