import os
import os.path as osp
import xml.etree.ElementTree as ET


def show_dir():
    dir_list = [a for a in os.listdir("./") if os.path.isdir(a) and a != 'test']
    total = 0
    categories = {}
    for dire in dir_list:
        jpeg_path = osp.join(dire, 'Annotations')
        xml_list = os.listdir(jpeg_path)
        total += len(xml_list)
        categories[dire] = len(xml_list)
    list1 = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    for i, (directory, num) in enumerate(list1):
        print(directory, num - 2680)
    print(total)


def merge_test():
    dir_list = [a for a in os.listdir("./") if os.path.isdir(a) and a != 'test']
    for dir_name in dir_list:
        Anno_path = osp.join(dir_name, "Annotations")
        Jpeg_path = osp.join(dir_name, "JPEGImages")
        Imag_path = osp.join(dir_name, "ImageSets", "Main")
        if not osp.exists(Imag_path):
            os.makedirs(Imag_path)
        test_anno_path = osp.join("test", "Annotations")
        test_jpeg_path = osp.join("test", "JPEGImages")
        test_txt_path = osp.join("test", "ImageSets", "Main", "test.txt")
        train_txt = open(osp.join(Imag_path, "train.txt"), 'w')
        valid_txt = open(osp.join(Imag_path, "valid.txt"), 'w')
        test_txt = open(osp.join(Imag_path, "test.txt"), 'w')
        anno_list = os.listdir(Anno_path)
        for anno_name in anno_list:
            anno_name = anno_name.replace(".xml", "\n")
            train_txt.write(anno_name)
        os.system("cp {}/* {}".format(test_anno_path, Anno_path))
        os.system("cp {}/* {}".format(test_jpeg_path, Jpeg_path))
        os.system("cp {} {}".format(test_txt_path, Imag_path))
        os.system("cp {} {}".format(test_txt_path, osp.join(Imag_path, 'valid.txt')))


def make_txt():
    dir_list = [a for a in os.listdir("./") if os.path.isdir(a) and a != 'test']
    for dir_name in dir_list:
        Anno_path = osp.join(dir_name, "Annotations")
        Imag_path = osp.join(dir_name, "ImageSets", "Main")
        ftest = open(osp.join(Imag_path, "test.txt"), 'r').readlines()
        ftrain = open(osp.join(Imag_path, "train.txt"), 'w')
        annos = os.listdir(Anno_path)
        for anno in annos:
            anno = anno.replace(".xml", "\n")
            if anno not in ftest:
                ftrain.write(anno)
