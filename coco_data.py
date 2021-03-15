import os,sys,re
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils import data as data
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
import pycocotools
from pycocotools.coco import COCO
import skimage.io as io
from config import Constants, Hyper

# dataset interface takes the ids of the COCO classes
class COCOData(data.Dataset):
    def __init__(self, **kwargs):
        self.stage = kwargs['stage']
        self.coco_classes_ids = kwargs['coco_classes_idx']
        self.adjusted_idx = kwargs['adjusted_classes_idx']
        self.coco_interface = kwargs['coco_interface']
        # this returns the list of image objects, equal to the number of images of the relevant class(es)
        self.datalist = kwargs['datalist'] 
        # load the list of the image
        self.img_data = self.coco_interface.loadImgs(self.datalist)

    # this method normalizes the image and converts it to Pytorch tensor
    # Here we use pytorch transforms functionality, and Compose them together,
    def transform(self, img):
        # these mean values are for RGB!!
        t_ = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            #transforms.Normalize(mean=[0.485, 0.457, 0.407],
                            #                     std=[1,1,1])
                            ])


        img = t_(img)
        # need this for the input in the model
        # returns image tensor (CxHxW)
        return img

    # downloadthe image 
    # return rgb image
    def load_img(self, idx): 
        coco_url = self.img_data[idx]['coco_url']
        im = np.array(io.imread(coco_url))
        im = self.transform(im)
        return im

    def load_label(self, idx): 
        # extract the id of the image, get the annotation ids
        im_id = self.img_data[idx]['id']
        annIds = self.coco_interface.getAnnIds(imgIds = im_id, catIds = self.coco_classes_ids, iscrowd=None)  
        # get the annotations 
        anns = self.coco_interface.loadAnns(annIds)         
        boxes = []
        ids = []
        # loop through all objects in the image
        # append id, bbox, extract mask and append it too
        for a in anns:
            adjusted_id = self.adjusted_idx[a['category_id']] 
            ids.append(adjusted_id)             
            box_coco = a['bbox']
            box = [box_coco[0], box_coco[1], box_coco[0]+box_coco[2], box_coco[1]+box_coco[3]]
            boxes.append(box)             
        # Careful with the data types!!!!
        # Also careful with the variable names!!!
        # If you accidentally use the same name for the object labels and the labs (output of the method) 
        # you get an infinite recursion
        boxes = torch.as_tensor(boxes, dtype = torch.float)
        ids = torch.tensor(ids, dtype=torch.int64)
        labs = {}
        labs['boxes'] = boxes
        labs['labels'] = ids
        return labs

    # number of images
    def __len__(self):
        return len(self.datalist)

    # return image + label 
    def __getitem__(self, idx):
         X = self.load_img(idx)
         y = self.load_label(idx) 
         return X,y

    def test_interface_with_single_image(self, image_id):
        ann_ids = self.coco_interface.loadAnns(image_id)
        img = self[image_id]
        self.coco_interface.showAnns(ann_ids)
        image = img.squeeze().permute(1,2,0)
        plt.imshow(image)
        plt.savefig("test.png")
        print("Image saved to test.png.")


if __name__ == "__main__":
    file_path = os.path.join(Constants.data_folder, Constants.images_train_file)
    coco_interface = COCO(file_path)
    selected_ann_ids = coco_interface.getImgIds()
    ####################################################################
    # load ids of images
    # Dataset class takes this list as an input and creates data objects 
    ann_ids = coco_interface.getAnnIds(imgIds=selected_ann_ids)
    ####################################################################
    # selected class ids: extract class id from the annotation
    coco_data_args = {'datalist':ann_ids, 'coco_interface':coco_interface, 'coco_ann_idx':selected_ann_ids, 'stage':'train'}
    coco_data = COCOData(**coco_data_args)
    coco_data.test_interface_with_single_image(59)

