import numpy as np

def get_negative_sample(attr_vec,topk=5):
    DialogeMask = np.zeros((64,64))
    for i in range(64):
        DialogeMask[i,i] = 2
    
    bz = len(attr_vec)
    attr_vec = attr_vec/(np.sqrt(np.square(attr_vec).sum(axis=-1)).reshape((-1,1))+10**(-6))
    attr_sim = np.dot(attr_vec,attr_vec.T)- DialogeMask[:bz,:bz]
    
    negative_sample_index = (-attr_sim).argsort(axis=-1)[:,:topk]
    for i in range(bz):
        ix = np.random.randint(topk)
        negative_sample_index[i,0] = negative_sample_index[i,ix]
    negative_sample_index = negative_sample_index[:,0]
    return negative_sample_index

def FairVFL_train(train_data,train_label,train_attr,model,rep_model,gender_model,age_model, gender_mapper, age_mapper, gender_cons_adv, age_cons_adv, bz=32):
    cons_att_label = np.zeros((64,2))
    cons_att_label[:,0] = 1

    random_index = np.random.permutation(len(train_data[0]))
    N = int(np.ceil(len(random_index)/bz))
    for i in range(N):
        s,e = i*bz,(i+1)*bz
        e = min(e,len(random_index))
        rbz = e - s
        sample_index = random_index[s:e]
        
        batch_sample = [train_data[i][sample_index] for i in range(len(train_data))]
        batch_label = [train_label[sample_index]] + [-train_attr[i][sample_index] for i in range(len(train_attr))]
                
        data_rep = rep_model.predict(batch_sample,verbose=False)
        
        gender_vec = gender_mapper.predict(data_rep)
        gender_neg_index = get_negative_sample(gender_vec,rbz)
        gender_pos = data_rep
        gender_neg = data_rep[gender_neg_index]
        gender_cons_adv[0].trainable = True
        gender_cons_adv[0].train_on_batch([gender_vec,gender_pos,gender_neg],cons_att_label[:rbz])
        gender_cons_adv[0].trainable = False
        gender_cons_adv[1].train_on_batch([data_rep,gender_pos,gender_neg],-cons_att_label[:rbz])
        
        age_vec = age_mapper.predict(data_rep)
        age_neg_index = get_negative_sample(age_vec,rbz)
        age_pos = data_rep
        age_neg = data_rep[age_neg_index]
        age_cons_adv[0].trainable = True
        age_cons_adv[0].train_on_batch([age_vec,age_pos,age_neg],cons_att_label[:rbz])
        age_cons_adv[0].trainable = False
        age_cons_adv[1].train_on_batch([data_rep,age_pos,age_neg],-cons_att_label[:rbz])
        
        
        gender_model.trainable = True
        gender_label = train_attr[0][sample_index]
        gender_model.train_on_batch(data_rep,gender_label)
        gender_model.train_on_batch(data_rep,gender_label)
        gender_model.trainable = False
        
        age_model.trainable = True
        age_label = train_attr[1][sample_index]
        age_model.train_on_batch(data_rep,age_label)
        age_model.train_on_batch(data_rep,age_label)
        age_model.trainable = False
        
        model.train_on_batch(batch_sample,batch_label)

