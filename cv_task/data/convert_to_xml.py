import os, sys
import glob
from PIL import Image
import json
from shutil import copyfile
# ICDAR image path

def convert(src_img_dir, src_txt_dir, ImageSets):

    img_Lists = glob.glob(src_img_dir + '/*.jpg')

    img_basenames = []  # e.g. 100.jpg
    for item in img_Lists:
        img_basenames.append(os.path.basename(item))

    img_names = []  # e.g. 100
    for item in img_basenames:
        temp1, temp2 = os.path.splitext(item)
        img_names.append(temp1)
    train_label = json.load(open(src_txt_dir, 'r'))
    txt_file = open(ImageSets, "w")
    for item in train_label:
        img = item['image_id']
        txt_file.write(img + "\n")
        # device_1_id = item['device_1_id']
        # device_2_id = item['device_2_id']
        items = item['items']

    # for img in img_names:
        img_path = src_img_dir + '/' + img + '.jpg'
        im = Image.open(img_path)
        width, height = im.size
        # open the crospronding txt file
        # gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()

        # write in xml file
        # os.mknod(src_txt_dir + '/' + img + '.xml')
        xml_dir = os.path.join(ann_dir)
        os.makedirs(xml_dir, exist_ok=True)
        xml_path = os.path.join(xml_dir, img + '.xml')
        xml_file = open(xml_path, 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('\t<folder>simple</folder>\n')
        xml_file.write('\t<filename>' + str(img) + '.jpg' + '</filename>\n')
        xml_file.write('\t<source>\n')
        xml_file.write('\t\t<database>' + 'The simple Database' + '</database>\n')
        xml_file.write('\t\t<annotation>' + 'simple' + '</annotation>\n')
        xml_file.write('\t\t<image>flickr</image>\n')
        xml_file.write('\t\t<flickrid>325991873</flickrid>\n')
        xml_file.write('\t</source>\n')
        xml_file.write('\t<owner>\n')
        xml_file.write('\t\t<flickrid>archin</flickrid>\n')
        xml_file.write('\t\t<name>?</name>\n')
        xml_file.write('\t</owner>\n')
        xml_file.write('\t<size>\n')
        xml_file.write('\t\t<width>' + str(width) + '</width>\n')
        xml_file.write('\t\t<height>' + str(height) + '</height>\n')
        xml_file.write('\t\t<depth>3</depth>\n')
        xml_file.write('\t</size>\n')
        xml_file.write('\t<segmented>0</segmented>\n')
        # write the region of text on xml file
        for img_each_label in items:
            cls = img_each_label['class']
            bbox = img_each_label['bbox']
            xml_file.write('\t<object>\n')
            xml_file.write('\t\t<name>{}</name>\n'.format(cls))
            xml_file.write('\t\t<pose>Unspecified</pose>\n')
            xml_file.write('\t\t<truncated>0</truncated>\n')
            xml_file.write('\t\t<difficult>0</difficult>\n')
            xml_file.write('\t\t<bndbox>\n')
            xml_file.write('\t\t\t<xmin>' + str(bbox[0]) + '</xmin>\n')
            xml_file.write('\t\t\t<ymin>' + str(bbox[1]) + '</ymin>\n')
            xml_file.write('\t\t\t<xmax>' + str(bbox[2]) + '</xmax>\n')
            xml_file.write('\t\t\t<ymax>' + str(bbox[3]) + '</ymax>\n')
            xml_file.write('\t\t</bndbox>\n')
            xml_file.write('\t</object>\n')

        xml_file.write('</annotation>')
        xml_file.flush()
        xml_file.close()
        # os.makedirs(os.path.join(src_ann_dir, 'street_5', device_2_id), exist_ok=True)

        copyfile(img_path, os.path.join(JPEGImages, img + '.jpg'))

src_img_dir = "Street_Dataset/Images"
# ICDAR ground truth path
src_txt_dir = "Street_Dataset/train_label.json"
ann_dir = "VOCdevkit/VOC2007/Annotations"
JPEGImages = "VOCdevkit/VOC2007/JPEGImages"
ImageSets = "VOCdevkit/VOC2007/ImageSets/Main"
os.makedirs(ann_dir, exist_ok=True)
os.makedirs(JPEGImages, exist_ok=True)
os.makedirs(ImageSets, exist_ok=True)
convert(src_img_dir, src_txt_dir, os.path.join(ImageSets, "train.txt"))
copyfile(os.path.join(ImageSets, "train.txt"), os.path.join(ImageSets, "trainval.txt"))
src_txt_dir = "Street_Dataset/test_label.json"
convert(src_img_dir, src_txt_dir, os.path.join(ImageSets, "test.txt"))