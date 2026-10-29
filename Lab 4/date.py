import pandas as pd
df = pd.DataFrame({ 'Date' : ['2023-01-01', '2023/02/15', 'Jan 5, 2025', '2023.04.05'] })
print("Original Data:\n", df)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
print(df)