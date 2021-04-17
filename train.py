import torch as T
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
from torch.utils import data
from config import Hyper, Constants, Global_Variable
from coco_data import COCO, COCOData
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils import load_checkpoint, save_checkpoint


def train():
    ###################### load COCO interface, the input is a json file with annotations ####################
    file_path = os.path.join(Constants.data_folder, Constants.images_train_file)
    coco_interface = COCO(file_path)
    # all indices of categories in MS COCO:
    all_cats = coco_interface.getCatIds()
    # add background class
    all_cats.insert(0, 0)
    print(all_cats, len(all_cats))
    # get names of cateogories
    # all_names = coco_interface.loadCats(all_cats[1:])
    # print(all_names)
    # choose the categories you want to work with
    # VERY CAREFUL WITH THIS LIST! SOME CLASSES ARE MISSING, TO TRAIN THE MODEL
    # YOU NEED TO ADJUST THE CLASS ID!!!
    selected_class_ids = coco_interface.getCatIds(catNms=[Constants.selected_category])
    adjusted_class_ids = {}
    for id, cl in enumerate(all_cats):
        adjusted_class_ids[cl] = id
    print("ADJUSTED CLASS IDS:")
    print(adjusted_class_ids)
    ####################################################################
    # load ids of images with this class
    # Dataset class takes this list as an input and creates data objects 
    im_ids = coco_interface.getImgIds(catIds=selected_class_ids)
    ####################################################################
    # selected class ids: extract class id from the annotation
    coco_data_args = {'datalist': im_ids, 'coco_interface': coco_interface, 'coco_classes_idx': selected_class_ids,
                      'stage': 'train', 'adjusted_classes_idx': adjusted_class_ids}
    coco_data = COCOData(**coco_data_args)
    coco_dataloader_args = {'batch_size': Hyper.batch_size, 'shuffle': True}
    coco_dataloader = data.DataLoader(coco_data, **coco_dataloader_args)
    step = 0
    # initilze model, loss, etc
    # , aux_logits = False
    fasterrcnn_args = {'num_classes':91, 'min_size':512, 'max_size':800}
    fasterrcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,**fasterrcnn_args)
    print(fasterrcnn_model)
    fasterrcnn_model = fasterrcnn_model.to(Constants.device)
    fasterrcnn_optimizer_pars = {'lr': Hyper.learning_rate}
    fasterrcnn_optimizer = optim.Adam(list(fasterrcnn_model.parameters()), **fasterrcnn_optimizer_pars)
    #####################################################################
    if Constants.load_model:
        epoch = load_checkpoint(fasterrcnn_model, fasterrcnn_optimizer)
    else:
        epoch = 0
    total_steps = 0
    step_loss = 0
    for _ in range(Hyper.total_epochs):
        fasterrcnn_model.train()  # Set model to training mode
        Global_Variable.is_train = True
        epoch += 1
        epoch_loss = 0
        start_time = time.strftime('%Y/%m/%d %H:%M:%S')
        print(f"{start_time} Starting epoch: {epoch}")
        step = 0
        for _, b in enumerate(coco_dataloader):
            fasterrcnn_optimizer.zero_grad()
            X,y = b
            step += 1
            total_steps += 1
            if step % 100 == 0:
                curr_time = time.strftime('%Y/%m/%d %H:%M:%S')
                print(f"-- {curr_time} epoch {epoch}, step: {step}, loss: {step_loss}")
            if Constants.device==T.device('cuda'):
                X = X.to(Constants.device)
                y['labels'] = y['labels'].to(Constants.device)
                y['boxes'] = y['boxes'].to(Constants.device)
            images = [im for im in X]
            targets = []
            lab = {'boxes': y['boxes'].squeeze_(0), 'labels': y['labels'].squeeze_(0)}
            # THIS IS IMPORTANT!!!!!
            # get rid of the first dimension (batch)
            # IF you have >1 images, make another loop
            # REPEAT: DO NOT USE BATCH DIMENSION 
            # Pytorch is sensitive to formats. Labels must be int64, bboxes float32
            lab['boxes'] = y['boxes'].squeeze_(0)
            lab['labels'] = y['labels'].squeeze_(0)
            targets.append(lab)
            is_bb_degenerate = check_if_target_bbox_degenerate(targets)
            if is_bb_degenerate:
                continue  # Ignore images with degenerate bounding boxes
            # avoid empty objects
            if len(targets) > 0:
                loss = fasterrcnn_model(images, targets)
                total_loss = 0
                for k in loss.keys():
                    total_loss += loss[k]

                step_loss = total_loss.item()
                epoch_loss += total_loss.item()
                total_loss.backward()
                fasterrcnn_optimizer.step()
        epoch_loss = epoch_loss / len(coco_dataloader)
        print(f"Loss in epoch {epoch} = {epoch_loss}")
        fasterrcnn_model.eval()
        Global_Variable.is_train = False
        if Constants.save_model:
            print(f"Save model for epoch {epoch}")
            checkpoint = {'epoch': epoch, 
                        "model_state_dict": fasterrcnn_model.state_dict(), 
                        "optimizer_state_dict": optim.state_dict()
                        "loss": epoch_loss}
            save_checkpoint(checkpoint)

    end_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print(f"Training end time: {end_time}")

def check_if_target_bbox_degenerate(targets):

    if targets is None:
        return False

    for target_idx, target in enumerate(targets):
        boxes = target["boxes"]
        degenerate_boxes = None
        if len(boxes.shape) != 2:
            return True

        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
            # print the first degenerate box
            bb_idx = T.where(degenerate_boxes.any(dim=1))[0][0]
            degen_bb = boxes[bb_idx].tolist()
            print("All bounding boxes should have positive height and width.")
            print(f"Found invalid box {degen_bb} for target at index {target_idx}.")
            return True
    return False


if __name__ == "__main__":
    train()
