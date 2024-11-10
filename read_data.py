import glob
import numpy as np
from os.path import join
import pickle


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def gather_data(dir_pattern):
    train_files = glob.glob(dir_pattern)
    first_batch = unpickle(train_files[0])
    images, labels = first_batch["data".encode()], first_batch["labels".encode()]
    for ind in range(1, len(train_files)):
        batch_data = unpickle(train_files[ind])
        images = np.vstack((images, batch_data["data".encode()]))
        labels.extend(batch_data["labels".encode()])
    twoD_labels = np.array(labels)[np.newaxis, ...].T
    return images, twoD_labels


if __name__ == "__main__":
    train_file = join("cifar-10-batches-py", "data_batch_*")
    test_file = join("cifar-10-batches-py", "test_batch")
    train_imgs, train_labels = gather_data(train_file)
    test_imgs, test_labels = gather_data(test_file)
    np.savez("train_data.npz", images=train_imgs, labels=train_labels)
    np.savez("test_data.npz", images=test_imgs, labels=test_labels)
