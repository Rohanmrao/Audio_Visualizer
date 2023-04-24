import pandas as pd

import re
#input the genre file 
df=pd.read_csv("classical.csv")
#df.head()

#extracted list of rows 
df1= df.iloc[::8].copy()
#df1

df1['rgb_values'] = df1['RGB'].apply(lambda x: re.findall(r'\d+', x)).apply(lambda x: [int(val) for val in x]).apply(lambda x: x[:-2])
#df1

df1.drop(["RGB"], axis=1 , inplace=True)
print(df1.count())
my_list=df1.values.tolist()
extracted_list = [sub_list[0] for sub_list in my_list]
nested_tuple = tuple(tuple(rgb) for rgb in extracted_list)
print(nested_tuple)