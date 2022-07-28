import json
import os

import numpy as np
import torch
import torch.utils.data as data

import preprocessing_captions
import preprocessing_image
import cv2


def create_list(annotation_file, img_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    imgList = list()
    for f in os.listdir(img_file):
        idx = int(f[:12])
        captions = ["<start> " + annotations['annotations'][i]['caption'] + " <end>" for i in
                    range(len(annotations['annotations'])) if
                    annotations['annotations'][i]["image_id"] == idx]
        imgList.append([f, captions])
    return imgList


class Dataset(data.Dataset):
    def __init__(self, input_path, output_path, annotation_file, max_qst_length):
        preprocessing_image.preprocessing_images(input_path=input_path, output_path=output_path)
        data_list = create_list(annotation_file=annotation_file, img_file=output_path)
        self.data_list, self.num_words = preprocessing_captions.preprocessing_captions(data_list, max_qst_length)
        self.max_qst_length = max_qst_length
        self.output_path = output_path
        self.input_path = input_path

    def __getitem__(self, item):
        image = cv2.imread(os.path.join(self.output_path, self.data_list[item][0]))
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32) / 256

        idx = torch.randint(0, len(self.data_list[item][1]), (1,)).item()
        caption = self.data_list[item][1][idx]
        caption = torch.tensor(caption)

        sample = (image, caption)
        return sample

    def __len__(self):
        return len(self.data_list)


def get_loader(input_path="./val2017", output_path="./resized_val2017",
               annotation_file='./annotations_trainval2017/annotations/captions_val2017.json', max_qst_length=30, batch_size=5):
    dataset = Dataset(input_path=input_path, output_path=output_path, annotation_file=annotation_file,
                      max_qst_length=max_qst_length)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader
