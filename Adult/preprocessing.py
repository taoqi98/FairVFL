import os
import numpy as np

class FeatureDictClass:
    def __init__(self,):
        self.word_dict = {}
        self.word_index = 1
    def put(self,f):
        if not f in self.word_dict:
            self.word_dict[f] = self.word_index
            self.word_index += 1
    def get(self,f):
        if f in self.word_dict:
            return self.word_dict[f]
        return 0

def load_data(data_root_path,val_ratio = 0.2):

    with open(os.path.join(data_root_path,'adult.data')) as f:
        train_lines = f.readlines()
    train_lines.remove('\n')
    with open(os.path.join(data_root_path,'adult.test')) as f:
        test_lines = f.readlines()
    test_lines = test_lines[1:]
    test_lines.remove('\n')
    
    lines = train_lines + test_lines
    train_index = len(train_lines)
    
    Attributes = {'age':[],'gender':[]}
    Labels = []
    Features = []
    
    Arr = []

    for i in range(len(lines)-1):
        splited = lines[i].strip('\n').split(', ')
        age = int(splited[0])
        gender = splited[-6]
        if gender == 'Male':
            gender = 1
        else:
            gender = 0

        if age<=24:
            age = 0
        elif age <=34:
            age = 1
        elif age <=49:
            age = 2
        else:
            age = 3


        Attributes['age'].append(age)
        Attributes['gender'].append(gender)

        label = splited[-1]
        if '>' in label:
            label = [0,1]
        else:
            label = [1,0]

        Labels.append(label)

        feature = []
        for i in range(len(splited)):
            if i in [0,9,14]:
                continue
            if i == 2:
                splited[i] = int(splited[i])
            feature.append(splited[i])
        Features.append(feature)
        Arr.append(feature[1])
    
    Arr = np.array(Arr)
    Avg = Arr.mean()
    Std = Arr.std()
    
    FeatureDict = {}
    for fid in range(len(feature)):
        FeatureDict[fid] = FeatureDictClass()
        for i in range(len(Features)):
            f = Features[i][fid]
            if fid == 1:
                f = (f-Avg)/Std
                f = min(f,2)
                f = max(f,-1.5)
                f = (f+1.5)/3.5
                f = 100*f
                f = int(f)
                
            FeatureDict[fid].put(f)
            Features[i][fid] = FeatureDict[fid].get(f)
            
    for attr in Attributes:
        Attributes[attr] = np.array(Attributes[attr])
    Labels = np.array(Labels)
    Features = np.array(Features)
    
    TrainFeatures = Features[:train_index]
    TestFeatures = Features[train_index:]
    TrainLabels = Labels[:train_index]
    TestLabels = Labels[train_index:]
    TrainAttri = {}
    TestAttri = {}
    
    for attr in Attributes:
        TrainAttri[attr] = Attributes[attr][:train_index]
        TestAttri[attr] = Attributes[attr][train_index:]
    
    random_index = np.random.permutation(len(TrainFeatures))
    val_num = int(val_ratio*len(random_index))
    train_index = random_index[val_num:]
    val_index = random_index[:val_num]
    
    ValFeatures = TrainFeatures[val_index]
    TrainFeatures = TrainFeatures[train_index]
    
    ValLabels = TrainLabels[val_index]
    TrainLabels = TrainLabels[train_index]
    
    ValAttri = {}
    for attr in Attributes:
        ValAttri[attr] = TrainAttri[attr][val_index]
        TrainAttri[attr] = TrainAttri[attr][train_index]
    
    LabelWeights = {0:TrainLabels[:,1].mean(),1:1-TrainLabels[:,1].mean()}
        
    return LabelWeights, FeatureDict, TrainFeatures, ValFeatures, TestFeatures, TrainAttri, ValAttri, TestAttri, TrainLabels, ValLabels, TestLabels