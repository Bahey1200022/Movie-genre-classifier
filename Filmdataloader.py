import csv
import ast 
import pandas as pd
class FilmdataLoader():
    def __init__(self,csv_path,poster_dir):
        self.data=list()
        self.poster=poster_dir
        df=pd.read_csv(csv_path, index_col=False)
        for row in df.iterrows():
            genre_list = ast.literal_eval(row[1]['genres'])
            self.data.append((row[1]['path'],genre_list))
        
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self,index):
        return self.data[index]
       
                
        
        