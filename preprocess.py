import pandas as pd
csv="final.csv"
df=pd.read_csv("output.csv")
ids=pd.read_csv("ids.csv")
import ast
for i,row in enumerate(df.iterrows()):
    genres_list = df.iloc[i, 1]
    genre_list = ast.literal_eval(genres_list)  # convert string to list
    for genre in genre_list:
        print(ids.iloc[:,0][genre])
    
    



