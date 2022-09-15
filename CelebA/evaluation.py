from sklearn.metrics import f1_score
from models import get_eval_classifier
import json
import numpy as np 

def eval_fairness(ATTR_INDEX,test_data_reps,test_attr):
    
    N = int(0.5*len(test_data_reps))

    attr_classifier = get_eval_classifier()
    for i in range(6):
        attr_classifier.fit(test_data_reps[:N],test_attr[ATTR_INDEX][:N],verbose=False)
        attr_pred = attr_classifier.predict(test_data_reps[N:]).argmax(axis=-1)
        acc = f1_score(test_attr[ATTR_INDEX][N:].argmax(axis=-1),attr_pred,average='micro')
        f1 = f1_score(test_attr[ATTR_INDEX][N:].argmax(axis=-1),attr_pred,average='macro')
        #print(acc,f1)
    return f1


def evaluation(model,rep_model,test_data,test_label,test_attr,):
    
    pred = model.predict(test_data)[0].argmax(axis=-1)
    acc = f1_score(test_label.argmax(axis=-1),pred,average='micro')
    f1 = f1_score(test_label.argmax(axis=-1),pred,average='macro')
    
    test_data_reps = rep_model.predict(test_data) 
    gender_f1 = eval_fairness(0,test_data_reps,test_attr)
    age_f1 = eval_fairness(1,test_data_reps,test_attr)
        
    test_result = [acc,f1,gender_f1,age_f1]
    test_result = np.array(test_result)
    
    return test_result

def DumpResult(Saved_Results,Result,topk=2,filename='FairVFL.json'):
    Result = np.array(Result)
    test_result = Result[:,0]
    val_result  = Result[:,1]
    valid_epoch = np.where(val_result[:,0]>0.8)[0]
    if len(valid_epoch)<6:
        return 

    test_result = test_result[valid_epoch]
    val_result  = val_result[valid_epoch]
    inxs = (val_result[:,2].argsort())[:topk]
    inx = inxs[np.random.randint(0,topk)]
    test_result = test_result[inx]

    Saved_Results.append(test_result)
    with open(filename,'a') as f:
        s = json.dumps(test_result.tolist()) + '\n'
        f.write(s)
    
    return 