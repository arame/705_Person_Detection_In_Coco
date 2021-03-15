import torch as T
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import time
from torch.utils import data
from tqdm import tqdm
from config import Hyper, Constants
from coco_data import COCO, COCOData
import os
from utils import load_checkpoint, save_checkpoint

def train():
    ###################### load COCO interface, the input is a json file with annotations ####################
    file_path = os.path.join(Constants.data_folder, Constants.images_train_file)
    coco_interface = COCO(file_path)
    # all indices of categories in MS COCO:
    all_cats = coco_interface.getCatIds()
    # add background class
    all_cats.insert(0,0)
    print(all_cats, len(all_cats))
    # get names of cateogories
    all_names = coco_interface.loadCats(all_cats[1:])
    #print(all_names)
    # choose the categories you want to work with
    # VERY CAREFUL WITH THIS LIST! SOME CLASSES ARE MISSING, TO TRAIN THE MODEL
    # YOU NEED TO ADJUST THE CLASS ID!!!
    selected_class_ids = coco_interface.getCatIds(catNms=[Constants.selected_category])
    adjusted_class_ids = {}
    for id, cl in enumerate(all_cats):
        adjusted_class_ids[cl] = id
    print ("ADJUSTED CLASS IDS:")
    print(adjusted_class_ids) 
    ####################################################################
    # load ids of images with this class
    # Dataset class takes this list as an input and creates data objects 
    im_ids = coco_interface.getImgIds(catIds=selected_class_ids)
    ####################################################################
    # selected class ids: extract class id from the annotation
    coco_data_args = {'datalist':im_ids, 'coco_interface':coco_interface, 'coco_classes_idx':selected_class_ids,'stage':'train', 'adjusted_classes_idx':adjusted_class_ids}
    coco_data = COCOData(**coco_data_args)
    coco_dataloader_args = {'batch_size':Hyper.batch_size, 'shuffle':True}
    coco_dataloader = data.DataLoader(coco_data, **coco_dataloader_args)
    step = 0
    # initilze model, loss, etc
    fasterrcnn_args = {'num_classes':81, 'min_size':512, 'max_size':800}
    fasterrcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,**fasterrcnn_args)
    print(fasterrcnn_model)
    fasterrcnn_model = fasterrcnn_model.to(Constants.device)
    fasterrcnn_optimizer_pars = {'lr':Hyper.learning_rate}
    fasterrcnn_optimizer = optim.Adam(list(fasterrcnn_model.parameters()),**fasterrcnn_optimizer_pars)
    #####################################################################
    if Constants.load_model:
        step = load_checkpoint(fasterrcnn_model, fasterrcnn_optimizer)

    fasterrcnn_model.train()   # Set model to training mode
    start_time = time.time()
    for epoch in range(Hyper.total_epochs):
        epoch_loss = 0 
        print(f"Epoch: {epoch + 1}")
        if Constants.save_model:
            checkpoint = {
                "state_dict": fasterrcnn_model.state_dict(),
                "optimizer": fasterrcnn_optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)


        # tqdm - Decorate an iterable object, 
        # returning an iterator which acts exactly like the original iterable, 
        # but prints a dynamically updating progressbar every time a value is requested.

        #for _, b in tqdm(enumerate(coco_dataloader), total=len(coco_dataloader), leave=False):
        for _, b in enumerate(coco_dataloader):
            fasterrcnn_optimizer.zero_grad()
            X,y = b
            if Constants.device==T.device('cuda'):
                X = X.to(Constants.device)
                y['labels'] = y['labels'].to(Constants.device)
                y['boxes'] = y['boxes'].to(Constants.device)
            images = [im for im in X]
            targets = []
            lab={}
            # THIS IS IMPORTANT!!!!!
            # get rid of the first dimension (batch)
            # IF you have >1 images, make another loop
            # REPEAT: DO NOT USE BATCH DIMENSION 
            # Pytorch is sensitive to formats. Labels must be int64, bboxes float32, masks uint8
            lab['boxes'] = y['boxes'].squeeze_(0)
            lab['labels'] = y['labels'].squeeze_(0)
            targets.append(lab)
            # avoid empty objects
            if len(targets)>0:
                loss = fasterrcnn_model(images, targets)
                total_loss = 0
                for k in loss.keys():
                    total_loss += loss[k]

                epoch_loss += total_loss.item()
                total_loss.backward()
                fasterrcnn_optimizer.step()
        epoch_loss = epoch_loss/len(coco_dataloader)
        print("Loss in epoch {0:d} = {1:.3f}".format(epoch, epoch_loss))
        fasterrcnn_model.eval()
        if Constants.save_model:
            print(f"Save model for epoch {epoch + 1}")
            save_checkpoint(fasterrcnn_model)

if __name__ == "__main__":
    train()