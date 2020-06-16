import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=['train', 'test']

classes = ["basket", "carton", "chair", "electrombile", "gastank", "sunshade", "table"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open('VOCdevkit/VOC2007/Annotations/%s.xml'%(image_id))
    out_file = open('VOCdevkit/VOC2007/labels/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for image_set in sets:
    os.makedirs('VOCdevkit/VOC2007/labels/', exist_ok=True)
    image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open('custom/%s.txt'%(image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC2007/JPEGImages/%s.jpg\n'%(wd, image_id))
        convert_annotation(image_id)
    list_file.close()
class_names = open('custom/classes.names', 'w')

for cls in classes:
    class_names.write(cls + '\n')
data_config = open("../config/custom.data", "w")
data_config.write("classes= 7\n")
data_config.write("train=%s\n"%(os.path.join(wd, "custom/train.txt")))
data_config.write("valid=%s\n"%(os.path.join(wd, "custom/test.txt")))
data_config.write("test=%s\n"%(os.path.join(wd, "custom/test.txt")))
data_config.write("names=%s\n"%(os.path.join(wd, "custom/classes.names")))
# os.system("cat 2007_train.txt > train.txt")
# os.system("cat 2007_test.txt > test.txt")

