import pandas as pd

df1 = pd.read_csv('data/StormEvents_details-ftp_v1.0_d2023_c20260323.csv.gz')
df2 = pd.read_csv('data/StormEvents_details-ftp_v1.0_d2024_c20260323 2.csv')


combined_df = pd.concat([df1, df2], ignore_index=True)


top_10 = combined_df['EVENT_TYPE'].value_counts().head(10)
print(top_10)
