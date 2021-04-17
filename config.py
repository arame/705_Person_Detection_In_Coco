import torch as T

class Hyper:
    total_epochs = 1
    learning_rate = 1e-6
    batch_size = 1
    is_url_read = False

    [staticmethod]   
    def display():
        print("The Hyperparameters")
        print("-------------------")
        print(f"NUmber of epochs = {Hyper.total_epochs}")
        print(f"learning rate = {Hyper.learning_rate}")
        print(f"batch_size = {Hyper.batch_size}")

class Constants:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    selected_category = 'person'
    load_model = False
    save_model = True
    backup_model_folder = "../backup_coco"
    backup_model_path = "../backup_coco/model.pth"
    data_folder = "../mscoco/annotations"
    images_train_file = "instances_train2017.json"
    images_val_file = "instances_val2017.json"
    train_data_folder = "../mscoco/train2017"
    val_data_folder = "../mscoco/val2017"

class Global_Variable:
    is_train = True

