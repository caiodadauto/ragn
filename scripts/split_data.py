import random
from os import listdir, rename, makedirs
from os.path import join


if __name__ == "__main__":
    val_ratio = 0.2
    train_ratio = 0.7
    random.seed(12345)
    file_names = listdir("data/")
    random.shuffle(file_names)
    makedirs(join("data", "val"))
    makedirs(join("data", "test"))
    makedirs(join("data", "train"))
    train_size = int(len(file_names) * train_ratio)
    val_size = int(len(file_names) * val_ratio)
    _ = [
        rename(join("data", name), join("data", "train", name))
        for name in file_names[:train_size]
    ]
    _ = [
        rename(join("data", name), join("data", "val", name))
        for name in file_names[train_size : (train_size + val_size)]
    ]
    _ = [
        rename(join("data", name), join("data", "test", name))
        for name in file_names[(train_size + val_size) :]
    ]
