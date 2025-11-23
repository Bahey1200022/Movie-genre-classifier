import csv
import ast 
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
import torch

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
  
])


class FilmdataLoader():
    def __init__(self,csv_path,poster_dir):
        self.data=list()
        self.poster=poster_dir
        df=pd.read_csv(csv_path, index_col=False)
        for row in df.iterrows():
            complete_path=os.path.join(poster_dir,row[1]['path'])
            genre_list = ast.literal_eval(row[1]['genres'])
            one_hot = torch.zeros(19)
            one_hot[genre_list] = 1
            self.data.append((complete_path,one_hot))
        
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self,index):
        image =Image.open(self.data[index][0]).convert("RGB")
        # image =preprocess(image).unsqueeze(0) 
        

        
        return (image,self.data[index][1])
       
                
        
        