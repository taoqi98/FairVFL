from sklearn.metrics import f1_score
from models import *
import numpy as np
from keras.utils.np_utils import to_categorical
import json

def evaluate_performance(model,LabelWeights,TestFeatures,TestLabels,TestAttri):
    pred = model.predict(TestFeatures)[0]
    
    acc = f1_score(TestLabels[:,1],pred[:,1]>LabelWeights[0],average='micro')
    f1 = f1_score(TestLabels[:,1],pred[:,1]>LabelWeights[0],average='weighted')
    
    return acc,f1

def evaluate_attribute(attr,data_rep,TestAttri):
    n = int(len(data_rep)*0.75)
    CateNum = {'age':4,'gender':2}

    R = []
    for i in range(2):
        fairness_attacker = get_fairness_attacker(CateNum[attr])
        fairness_attacker.fit(data_rep[:n],to_categorical(TestAttri[attr][:n]),epochs=5,verbose=False)
        pred = fairness_attacker.predict(data_rep[n:])
        r = f1_score(TestAttri[attr][n:],pred.argmax(axis=-1),average='macro')
        r = np.array(r)
        R.append(r)
    R = np.array(R)
    R = R.mean(axis=0)
    return R

def evaluate_attribute_val(attr,train_data_rep,val_data_rep,TrainAttri,ValAttri):
    CateNum = {'age':4,'gender':2}
    index = np.random.permutation(len(train_data_rep))[:len(train_data_rep)//2]
    R = []
    for i in range(2):
        fairness_attacker = get_fairness_attacker(CateNum[attr])
        fairness_attacker.fit(train_data_rep[index],to_categorical(TrainAttri[attr][index]),epochs=5,verbose=False)
        pred = fairness_attacker.predict(val_data_rep)
        r = f1_score(ValAttri[attr],pred.argmax(axis=-1),average='macro')
        r = np.array(r)
        R.append(r)
    R = np.array(R)
    R = R.mean(axis=0)
    return R

def attack_privacy(n,data_rep,feature_dict,feature):
    
    vec_input = Input(shape=(data_rep.shape[-1],))
    score = Dense(256,activation='relu')(vec_input)
    score = Dense(256,activation='relu')(score)
    logit = Dense(len(feature_dict.word_dict)+1,activation='softmax')(score)
    
    behavior_predictor = Model(vec_input,logit)
    
    behavior_predictor.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001), 
                  metrics=['acc'])
    
    for i in range(10):
        behavior_predictor.fit(data_rep[:n],to_categorical(feature[:n],len(feature_dict.word_dict)+1),verbose=0) #class_weight=weights)
    pred = behavior_predictor.predict(data_rep[n:])
    f1 = f1_score(feature[n:],pred.argmax(axis=-1),average='macro')
    
    return f1
    
def evaluate_privacy(rep_model,Mapper,TestFeatures,FeatureDict):
    
    data_rep = rep_model.predict(TestFeatures)
    protected_data_rep = Mapper['gender'].predict(data_rep)
    n = int(len(data_rep)*0.25)
    
    Privacy = {}

    for feature_id in [2,3,4,5,6,]:
        Privacy[feature_id] = []

        for i in range(5):
            f11 = attack_privacy(n,protected_data_rep,FeatureDict[feature_id],TestFeatures[:,feature_id],)
                        
            f12 = attack_privacy(n,data_rep,FeatureDict[feature_id],TestFeatures[:,feature_id],)

            random_preds = np.random.randint(1,len(FeatureDict[feature_id].word_dict)+1,size=(len(data_rep)-n,))
            f13 = f1_score(TestFeatures[n:,feature_id],random_preds,average='macro')
            
            Privacy[feature_id].append([f11,f12,f13])

        Privacy[feature_id] = np.array(Privacy[feature_id]).mean(axis=0)
        Privacy[feature_id] = Privacy[feature_id].tolist()
        
    return Privacy


def evaluate_val(model,rep_model,LabelWeights,TrainFeatures,TrainAttri,ValFeatures,ValLabels,ValAttri):

    performance = evaluate_performance(model,LabelWeights,ValFeatures,ValLabels,ValAttri)
    
    train_data_rep = rep_model.predict(TrainFeatures)
    val_data_rep = rep_model.predict(ValFeatures)
    gender = evaluate_attribute_val('gender',train_data_rep,val_data_rep,TrainAttri,ValAttri)
    age = evaluate_attribute_val('age',train_data_rep,val_data_rep,TrainAttri,ValAttri)
    
    result = {'performance':performance,'gender':gender,'age':age}
    
    return result

def evaluate_test(model,rep_model,LabelWeights,TestFeatures,TestLabels,TestAttri):
    
    performance = evaluate_performance(model,LabelWeights,TestFeatures,TestLabels,TestAttri)
    
    data_rep = rep_model.predict(TestFeatures)    
    gender = evaluate_attribute('gender',data_rep,TestAttri)
    age = evaluate_attribute('age',data_rep,TestAttri)
    
    result = {'performance':performance,'gender':gender,'age':age}
    
    return result


def dump_result(Result,filename='FairVFL.json',topk=2):
    
    test = []
    val = []
    privacy = {v:[] for v in [2,3,4,5,6]}
    for j in range(len(Result)):
        
        test.append([])
        val.append([])

        test[-1] += Result[j][0]['performance']
        test[-1].append(Result[j][0]['gender'])
        test[-1].append(Result[j][0]['age'])

        val[-1] += Result[j][1]['performance']
        val[-1].append(Result[j][1]['gender'])
        val[-1].append(Result[j][1]['age'])
        
        pri = Result[j][2]
        for fid in privacy:
            privacy[fid].append(pri[fid])
            
    test = np.array(test)
    val = np.array(val)

    valid_epoch = np.where(val[:,0]>0.70)[0]

#     if len(valid_epoch)<8:
#         return 
        
    valid_epoch = valid_epoch
    test = test[valid_epoch]
    val = val[valid_epoch]
    ixs = val[:,2:].sum(axis=-1).argsort()[:topk]
    ix = ixs[np.random.randint(min(topk,len(valid_epoch)))]

    test = test[ix].tolist()
    
    for fid in privacy:
        privacy[fid] = np.array(privacy[fid]).max(axis=0).tolist()
            
    s = json.dumps([test,privacy]) + '\n'
    with open(filename,'a') as f:
        f.write(s)