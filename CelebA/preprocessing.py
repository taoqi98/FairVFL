import os
import imageio
import numpy as np
from keras.utils.np_utils import to_categorical



def load_data(data_root_path,filename='zipped_images_4.npy'):
    images = np.load(os.path.join(data_root_path,filename))
    images = np.array(images,dtype='float32')
    images /= 255
    with open(os.path.join(data_root_path,'attribute.txt')) as f:
        lines = f.readlines()
    
    attribute_name = lines[1].strip().split()
    attributes = {attribute_name[i]:[] for i in range(len(attribute_name))}
    for i in range(2,len(lines)):
        values = lines[i].strip().split()[1:]
        for j in range(len(attribute_name)):
            n = attribute_name[j]
            v = (int(values[j])+1)//2
            attributes[n].append(v)
    for n in attributes:
        attributes[n] = np.array(attributes[n])
    return images, attributes


def data_partation(images, attributes,data_root_path,LABEL_NAME = 'Smiling', P1_NAME = ['Straight_Hair', 'Wavy_Hair']):
    with open(os.path.join(data_root_path,'eval.txt')) as f:
        lines = f.readlines()
    train_index = []
    val_index = []
    test_index = []
    for i in range(len(lines)):
        v = int(lines[i].strip().split()[1])
        if v == 0:
            train_index.append(i)
        elif v == 1:
            val_index.append(i)
        else:
            test_index.append(i)
            
    train_index = np.array(train_index)
    val_index = np.array(val_index)
    test_index = np.array(test_index)
    
    P1_attribute = np.concatenate([to_categorical(attributes[a]) for a in P1_NAME] ,axis=-1)
    
    train_data = [images[train_index],P1_attribute[train_index]] 
    train_label = to_categorical(attributes[LABEL_NAME][train_index])
    train_attr = [to_categorical(attributes['Male'][train_index]),to_categorical(attributes['Young'][train_index])]
    
    val_data = [images[val_index],P1_attribute[val_index]] 
    val_label = to_categorical(attributes[LABEL_NAME][val_index])
    val_attr = [to_categorical(attributes['Male'][val_index]),to_categorical(attributes['Young'][val_index])]
    
    test_data = [images[test_index],P1_attribute[test_index]]
    test_label = to_categorical(attributes[LABEL_NAME][test_index])
    test_attr = [to_categorical(attributes['Male'][test_index]),to_categorical(attributes['Young'][test_index])]
    
    
    return train_data,train_label,train_attr, val_data,val_label,val_attr, test_data,test_label,test_attr