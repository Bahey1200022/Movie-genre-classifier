import csv
class FilmdataLoader():
    def __init__(self,csv_path,poster_path):
        self.data=list()
        self.poster=poster_path
        with open("file.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            
            for row in reader :
                self.data.append((row[0],row[1]))
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self,index):
        return self.data[index]
       
                
        
        