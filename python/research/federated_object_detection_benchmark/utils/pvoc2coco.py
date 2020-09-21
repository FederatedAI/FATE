import os
import xml.etree.ElementTree as ET
import xmltodict
import json
from xml.dom import minidom
from collections import OrderedDict


#attrDict = {"images":[{"file_name":[],"height":[], "width":[],"id":[]}], "type":"instances", "annotations":[], "categories":[]}

#xmlfile = "000023.xml"


def generateVOC2Json(rootDir,xmlFiles, output_dir):
    attrDict = dict()
    #images = dict()
    #images1 = list()
    attrDict["categories"]=[{"supercategory":"none","id":1,"name":"header"},
                    {"supercategory":"none","id":2,"name":"row"},
                    {"supercategory":"none","id":3,"name":"logo"},
                    {"supercategory":"none","id":4,"name":"item_name"},
                {"supercategory":"none","id":5,"name":"item_desc"},
                {"supercategory":"none","id":6,"name":"price"},
                {"supercategory":"none","id":7,"name":"total_price_text"},
                {"supercategory":"none","id":8,"name":"total_price"},
                {"supercategory":"none","id":9,"name":"footer"}
                  ]
    images = list()
    annotations = list()
    for root, dirs, files in os.walk(rootDir):
        image_id = 0
        for file in xmlFiles:
            image_id = image_id + 1
            if file in files:
                
                #image_id = image_id + 1
                annotation_path = os.path.abspath(os.path.join(root, file))
                
                #tree = ET.parse(annotation_path)#.getroot()
                image = dict()
                #keyList = list()
                doc = xmltodict.parse(open(annotation_path).read())
                #print doc['annotation']['filename']
                image['file_name'] = str(doc['annotation']['filename'])
                #keyList.append("file_name")
                image['height'] = int(doc['annotation']['size']['height'])
                #keyList.append("height")
                image['width'] = int(doc['annotation']['size']['width'])
                #keyList.append("width")

                #image['id'] = str(doc['annotation']['filename']).split('.jpg')[0]
                image['id'] = image_id
                print("File Name: {} and image_id {}".format(file, image_id))
                images.append(image)
                # keyList.append("id")
                # for k in keyList:
                #     images1.append(images[k])
                # images2 = dict(zip(keyList, images1))
                # print images2
                #print images

                #attrDict["images"] = images

                #print attrDict
                #annotation = dict()
                id1 = 1
                if 'object' in doc['annotation']:
                    for obj in doc['annotation']['object']:
                        for value in attrDict["categories"]:
                            annotation = dict()
                            #if str(obj['name']) in value["name"]:
                            if str(obj['name']) == value["name"]:
                                #print str(obj['name'])
                                #annotation["segmentation"] = []
                                annotation["iscrowd"] = 0
                                #annotation["image_id"] = str(doc['annotation']['filename']).split('.jpg')[0] #attrDict["images"]["id"]
                                annotation["image_id"] = image_id
                                x1 = int(obj["bndbox"]["xmin"])  - 1
                                y1 = int(obj["bndbox"]["ymin"]) - 1
                                x2 = int(obj["bndbox"]["xmax"]) - x1
                                y2 = int(obj["bndbox"]["ymax"]) - y1
                                annotation["bbox"] = [x1, y1, x2, y2]
                                annotation["area"] = float(x2 * y2)
                                annotation["category_id"] = value["id"]
                                annotation["ignore"] = 0
                                annotation["id"] = id1
                                annotation["segmentation"] = [[x1,y1,x1,(y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]
                                id1 +=1

                                annotations.append(annotation)
                
                else:
                    print("File: {} doesn't have any object".format(file))
                #image_id = image_id + 1
                
            else:
                print("File: {} not found".format(file))
            

    attrDict["images"] = images    
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"

    #print attrDict
    jsonString = json.dumps(attrDict)
    with open(os.path.join(output_dr, "receipts_valid.json"), "w") as f:
        f.write(jsonString)

# rootDir = "/netscratch/pramanik/OBJECT_DETECTION/detectron/lib/datasets/data/Receipts/Annotations"
# for root, dirs, files in os.walk(rootDir):
#     for file in files:
#         if file.endswith(".xml"):
#             annotation_path = str(os.path.abspath(os.path.join(root,file)))
#             #print(annotation_path)
#             generateVOC2Json(annotation_path)
import sys

try:
    assert len(sys.argv) == 4
except:
    print("three arguments should be <trainfile.txt> <dataset_root> <output_dir>")
    exit(0)

trainFile = sys.argv[1]
trainXMLFiles = list()
with open(trainFile, "rb") as f:
    for line in f:
        fileName = line.strip()
        trainXMLFiles.append(fileName + ".xml")


rootDir = sys.argv[2]
output_dir = sys.argv[3]
generateVOC2Json(rootDir, trainXMLFiles, output_dir)
