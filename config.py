import torch as T

class Hyper:
    total_epochs = 1
    learning_rate = 1e-6
    batch_size = 1

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
    backup_model_folder = "../backup"
    backup_model_path = "../backup/model.pth"
    data_folder = "../mscoco/annotations"
    images_train_file = "instances_train2017.json"
    images_val_file = "instances_val2017.json"

