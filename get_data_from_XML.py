import numpy as np
import os
from xml.etree import ElementTree

class XML_preprocessor(object):

    def __init__(self, data_path):
        self.path_prefix = data_path
        self.num_classes = 5
        self.data = dict()
        self._preprocess_XML()

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            # print(self.path_prefix+filename)
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)/width
                    ymin = float(bounding_box.find('ymin').text)/height
                    xmax = float(bounding_box.find('xmax').text)/width
                    ymax = float(bounding_box.find('ymax').text)/height
                bounding_box = [xmin,ymin,xmax,ymax]
                bounding_boxes.append(bounding_box)
                class_name = object_tree.find('name').text
                one_hot_class = self._to_one_hot(class_name, filename)
                one_hot_classes.append(one_hot_class)
            image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            self.data[image_name] = image_data

    def _to_one_hot(self,name, filename):
        one_hot_vector = [0] * self.num_classes
        if name == 'crack':
            one_hot_vector[0] = 1
        elif name == 'carbonation':
            one_hot_vector[1] = 1
        elif name == 'free_lime_c':
            one_hot_vector[2] = 1
        elif name == 'free_lime_d':
            one_hot_vector[3] = 1
        elif name == 'free_lime_e':
            one_hot_vector[4] = 1
        else:
            print('unknown label: %s' %name)
            print(filename)

        return one_hot_vector

## example on how to use it
import pickle
data = XML_preprocessor('./datasets/annotation/').data
pickle.dump(data,open('datasets.pkl','wb'))
