import numpy as np
from keras.utils.np_utils import to_categorical

def train_model(model,rep_model,Discrimator,Mapper,Contrastive_Discrimator,Contrastive_Attacker,LabelWeights,TrainFeatures,TrainAttri,TrainLabels):
    bz = 64
    TOPK = {'gender':10,'age':5}

    DialogeMask = np.zeros((64,64))
    for i in range(64):
        DialogeMask[i,i] = 2
    ContDisLabel = np.zeros((64,2))
    ContDisLabel[:,0] = 1


    for i in range(len(TrainFeatures)//bz):
        start,ed = bz*i,bz*i+bz
        x = TrainFeatures[start:ed]
        y = [TrainLabels[start:ed],-to_categorical(TrainAttri['gender'][start:ed]),-to_categorical(TrainAttri['age'][start:ed])]
            
        data_rep = rep_model.predict(x,verbose=False)
        
        for attr in Mapper:
            
            topk = TOPK[attr]
            
            protected_rep = Mapper[attr].predict(data_rep,verbose=False)
            vec = protected_rep/np.sqrt(np.square(protected_rep).sum(axis=-1,keepdims=True)+10**(-6))
            s = np.dot(vec,vec.T) - DialogeMask[:bz,:bz]
            negative_sample_index = (-s).argsort(axis=-1)[:,:topk]
            for i in range(bz):
                ix = np.random.randint(topk)
                negative_sample_index[i,0] = negative_sample_index[i,ix]
            negative_sample_index = negative_sample_index[:,0]

            pos_data_rep = data_rep
            neg_data_rep = data_rep[negative_sample_index]

            Contrastive_Discrimator[attr].trainable = True
            Contrastive_Discrimator[attr].train_on_batch([protected_rep,pos_data_rep,neg_data_rep],ContDisLabel[:bz])
            Contrastive_Discrimator[attr].trainable = False
            
            Contrastive_Attacker[attr].train_on_batch([data_rep,pos_data_rep,neg_data_rep],-ContDisLabel[:bz])

            
        for attr in Discrimator:
            Discrimator[attr].trainable = True
            Discrimator[attr].train_on_batch(data_rep,to_categorical(TrainAttri[attr][start:ed]),)
            Discrimator[attr].train_on_batch(data_rep,to_categorical(TrainAttri[attr][start:ed]),)
            Discrimator[attr].trainable = False
                
        model.train_on_batch(x,y,class_weight=LabelWeights)