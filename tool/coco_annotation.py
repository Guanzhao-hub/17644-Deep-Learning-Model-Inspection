# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/08 11:45
@Author        : Tianxiaomo
@File          : coco_annotatin.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import json
from collections import defaultdict
from tqdm import tqdm
import os

"""hyper parameters"""
json_file_path = '../data/annotations/instances_train2017.json'
images_dir_path = './data/dataset/train/'
output_path = '../data/train.txt'
is_train = True

"""load json file"""
name_box_id = defaultdict(list)
id_name = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)

"""generate labels"""
images = data['images']
annotations = data['annotations']

image_slice = os.listdir("../data/dataset/train")
image_slice = ["./data/dataset/train"+"/"+img for img in image_slice]


for ant in tqdm(annotations):
    id = ant['image_id']
    # name = os.path.join(images_dir_path, images[id]['file_name'])
    name = os.path.join(images_dir_path, '{:012d}.jpg'.format(id))
    if name not in image_slice:
        continue
    cat = ant['category_id']

    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11

    name_box_id[name].append([ant['bbox'], cat])

print(len(name_box_id))
"""write to txt"""
with open(output_path, 'w') as f:
    for key in tqdm(name_box_id.keys()):
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
