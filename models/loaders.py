import numpy as np

path = "/home/major/Major-Project-Experiments/data/organizedPancreasData"

train_val_folder = "/train_np"
label_folder = "/label_np"
test_folder = "/test_np"

train_txt = "/home/major/Major-Project-Experiments/data/organizedPancreasData/train.txt"
test_txt = "/home/major/Major-Project-Experiments/data/organizedPancreasData/test.txt"
validation_txt = "/home/major/Major-Project-Experiments/data/organizedPancreasData/validation.txt"  

def get_image_and_label_from_npy(isTrain = False, is2D = False):
    """
    Load train or validation numpy arrays from corresponding npy files
    """
    image_np, label_np = [], []
    filename = train_txt if isTrain else validation_txt
    with open(filename) as file:
        for row in file:
            npy_file = row.strip('\n')
            np_arr = np.load(path + train_val_folder + '/' + npy_file)
            image_np.append(np_arr)
            np_arr = np.load(path + label_folder + '/' + npy_file)
            label_np.append(np_arr)
    return image_np, label_np

def get_test_np_from_npy(is2D = False):
    test_np = []
    with open(test_txt) as file:
        for row in file:
            npy_file = row.strip('\n')
            np_arr = np.load(path + test_folder + '/' + npy_file)
            test_np.append(np_arr)
    return test_np

x, y = get_image_and_label_from_npy()
print(f'Test: {len(x)} {len(y)}')