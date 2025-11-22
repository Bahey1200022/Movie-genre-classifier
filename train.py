from Filmdataloader import FilmdataLoader
import pandas as pd
import ast
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
from classifier import Classifier


x= torch.rand((1000,768))
f=Classifier(768,20)
print(f(x).shape)

