import torch as T
import torchvision.transforms as transforms
from PIL import Image
from config import Constants
import os

def check_folder(folder):
    if os.path.isdir(folder) == False:
        os.mkdir(folder)

def checkfolders():
    # Make sure the output folders exist
    check_folder(Constants.backup_model_folder)
    
def save_checkpoint(checkpoint):
    check_folder(Constants.backup_model_folder)
    print("=> Saving checkpoint")
    T.save(checkpoint, Constants.backup_model_path)

def load_checkpoint(model, optimizer):
    print("=> Loading checkpoint")

    checkpoint = T.load(Constants.backup_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return epoch