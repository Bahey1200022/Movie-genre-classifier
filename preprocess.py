import pandas as pd
import ast
csv="films.csv"
df=pd.read_csv("output.csv")
ids=pd.read_csv("ids.csv")
finaldf=pd.DataFrame(columns=['path','genres'])
for i, row in df.iterrows():
    genres_list = df.iloc[i, 1]
    genre_list = ast.literal_eval(genres_list)
    filmlist=list()
    for genre in genre_list:
        genre_id = ids.loc[ids['genre'] == genre, 'id'].values[0]
        filmlist.append(genre_id)
        
    finaldf.loc[i]=[row[0],filmlist]
    
finaldf.to_csv(csv,index=False)   

    
    



