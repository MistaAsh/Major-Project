# Copying the files in yolo format

import shutil
from tqdm import tqdm
import os

train_list = open('/home/major/Major-Project-Experiments/train_yolo.txt', 'r').read().strip().split('\n')
test_list = open('/home/major/Major-Project-Experiments/test_yolo.txt', 'r').read().strip().split('\n')

for train_label in tqdm(train_list):
    train_label_output = ('/').join(train_label.replace('preprocessed_labelsTr_bbox', 'final_yolo_dataset').split('/')[:-2]) + '/train/' + train_label.replace('preprocessed_labelsTr_bbox', 'final_yolo_dataset').split('/')[-1]
    shutil.copy(train_label, train_label_output)
    train_file = train_label.replace('/labels/', '/images/').replace('txt', 'jpeg')
    train_file_output = ('/').join(train_file.replace('preprocessed_labelsTr_bbox', 'final_yolo_dataset').split('/')[:-2]) + '/train/' + train_file.replace('preprocessed_labelsTr_bbox', 'final_yolo_dataset').split('/')[-1]
    shutil.copy(train_file, train_file_output)

for test_label in tqdm(test_list):
    test_label_output = ('/').join(test_label.replace('preprocessed_labelsTr_bbox', 'final_yolo_dataset').split('/')[:-2]) + '/val/' + test_label.replace('preprocessed_labelsTr_bbox', 'final_yolo_dataset').split('/')[-1]
    shutil.copy(test_label, test_label_output)
    test_file = test_label.replace('/labels/', '/images/').replace('txt', 'jpeg')
    test_file_output = ('/').join(test_file.replace('preprocessed_labelsTr_bbox', 'final_yolo_dataset').split('/')[:-2]) + '/val/' + test_file.replace('preprocessed_labelsTr_bbox', 'final_yolo_dataset').split('/')[-1]
    shutil.copy(test_file, test_file_output)