import pandas as pd

import re
#input the genre file 
df=pd.read_csv("classical.csv")


#extracted list of rows 
df1= df.iloc[::8].copy()

#extracting [r,g,b] values only using regex
df1['rgb_values'] = df1['RGB'].apply(lambda x: re.findall(r'\d+', x)).apply(lambda x: [int(val) for val in x]).apply(lambda x: x[:-2])


df1.drop(["RGB"], axis=1 , inplace=True)
print(df1.count()) # to check number of values 
my_list=df1.values.tolist() #converting dataframe to list
extracted_list = [sub_list[0] for sub_list in my_list]
nested_tuple = tuple(tuple(rgb) for rgb in extracted_list)# converting nested lists to nested tuples
print(nested_tuple)