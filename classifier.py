import torch 

class Classifier (torch.nn.Module):
    def __init__(self,input_dimensions,numClasses):
        super().__init__()
        self.l1=torch.nn.Linear(input_dimensions,512)
        self.l2=torch.nn.Linear(512,256)
        self.l3=torch.nn.Linear(256,32)
        self.l4=torch.nn.Linear(32,numClasses)
    def forward(self,x):
        x=self.l1(x)
        x=torch.nn.functional.relu(x)
        x=self.l2(x)
        x=torch.nn.functional.relu(x)
        x=self.l3(x)
        x=torch.nn.functional.relu(x)
        x=self.l4(x)
        x=torch.nn.functional.sigmoid(x)
        return x
        
        
            