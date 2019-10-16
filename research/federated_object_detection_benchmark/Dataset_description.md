## FedVision
**FedVison dataset** is created jointly by WeBank and ExtremeVision to facilitate the advancement of academic research 
and industrial applications of federated learning.

### The FedVision project

* Provides images data sets with standardized annotation for federated object detection.
* Provides key statistics  and systems metrics of the data sets.
* Provides a set of implementations of baseline for further research.

### Datasets
We introduce two realistic federated datasets.
 
* **Federated Street**, a real-world object detection dataset that annotates images captured by a set of street cameras 
based on object present in them, including 7 classes. In this dataset, each or every few cameras serve as a device.

 | Dataset | Number of devices | Total samples | Number of class| 
 |:---:|:---:|:---:|:---:|
 | Federated Street | 5, 20 | 956 | 7 |
 
### File descriptions

* **Street_Dataset.tar** contains the image data and ground truth for the train and test set of the street data set.
    * **Images**: The directory which contains the train and test image data.
    * **train_label.json**: The annotations file is saved in json format. **train_label.json** is a `list`, which 
    contains the annotation information of the Images set. The length of `list` is the same as the number of image and each value
    in the `list` represents one image_info. Each `image_info` is in format of `dictionary` with keys and values. The keys 
    of `image_info` are `image_id`, `device1_id`, `device2_id` and `items`. We split the street data set in two ways. For the first, we
    split the data into 5 parts according to the geographic information. Besides, we turn 5 into 20. Therefore we have `device1_id` and
     `device2_id`. It means that we have 5 or 20 devices. `items` is a list, which may contain multiple objects.  
    [  
     &emsp;    {  
     &emsp;&emsp;    `"image_id"`: the id of the train image, for example 009579.  
     &emsp;&emsp;    `"device1_id"`: the id of device1 ,specifies which device the image is on.   
     &emsp;&emsp;    `"device2_id"`: the id of device2.    
     &emsp;&emsp;    `"items"`: [  
     &emsp;&emsp;&emsp;       {  
     &emsp;&emsp;&emsp;&emsp;          `"class"`: the class of one object,  
     &emsp;&emsp;&emsp;&emsp;          `"bbox"`: ["xmin", "ymin", "xmax", "ymax"], the coordinates of a bounding box  
     &emsp;&emsp;&emsp;       },  
     &emsp;&emsp;&emsp;       ...  
     &emsp;&emsp;&emsp;       ]  
     &emsp;     },  
     &emsp;     ...  
    ]
    * **test_label.json**: The annotations of test data are almost the same as of the **train_label.json**. The only difference between them is that 
    the `image_info` of test data does not have the key `device_id`.  
   
### Evaluation
We use he standard [PASCAL VOC 2010](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/devkit_doc_08-May-2010.pdf) mean Average Precision (mAP) for evaluation (mean is taken over per-class APs).  
To be considered a correct detection, the overlap ratio ![avatar](http://fedcs.fedai.org.cn/1.png) between the predicted bounding box ![avatar](http://fedcs.fedai.org.cn/2.png) and ground truth bounding ![avatar](http://fedcs.fedai.org.cn/3.png) by the formula  
<a href="https://www.codecogs.com/eqnedit.php?latex=$$a_o=\frac{area(B_p&space;\cap&space;B_{g_t})}{area(B_p&space;\cup&space;B_{g_t})}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?$$a_o=\frac{area(B_p&space;\cap&space;B_{g_t})}{area(B_p&space;\cup&space;B_{g_t})}$$" title="$$a_o=\frac{area(B_p \cap B_{g_t})}{area(B_p \cup B_{g_t})}$$" /></a>  
when <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;$$B_p&space;\cap&space;B_{g_t}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{100}&space;$$B_p&space;\cap&space;B_{g_t}$$" title="$$B_p \cap B_{g_t}$$" /></a> denotes the intersection of the predicted and ground truth bounding boxes and <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;$$B_p&space;\cup&space;B_{g_t}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{100}&space;$$B_p&space;\cup&space;B_{g_t}$$" title="$$B_p \cup B_{g_t}$$" /></a> their union.
Average Precision is calculated for each class respectively.  
<a href="https://www.codecogs.com/eqnedit.php?latex=$$AP=\frac{P}{n}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?$$AP=\frac{P}{n}$$" title="$$AP=\frac{P}{n}$$" /></a>  
where n is the number of total object in given class.
For $k$ classes, mAP is the average of APs.    
<a href="https://www.codecogs.com/eqnedit.php?latex=$$mAP=\frac{\sum_{i=1}^{k}AP_i}{k}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?$$mAP=\frac{\sum_{i=1}^{k}AP_i}{k}$$" title="$$mAP=\frac{\sum_{i=1}^{k}AP_i}{k}$$" /></a>
