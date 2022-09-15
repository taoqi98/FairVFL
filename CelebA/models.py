from keras.layers import *
from keras.models import Model
from keras.optimizers import *

def get_discrimator():
    inputx = Input(shape=(512,))
    protected_vec = Dense(256,)(inputx)
    protected_vec = Dense(128,activation='relu')(protected_vec)
    protected_vec = Dense(128)(protected_vec)
    
    v = Dense(128,activation='relu')(protected_vec)
    v = Dense(128,activation='relu')(v)
    y = Dense(2,activation='softmax')(v)
    model =  Model(inputx,y)
    model.compile(loss=['categorical_crossentropy',],
                  optimizer=Adam(lr=0.0001), 
                  metrics=['acc'])
    
    mapper = Model(inputx,protected_vec)
    mapper.compile(loss=['categorical_crossentropy',],
                  optimizer=Adam(lr=0.0001), 
                  metrics=['acc'])    
    return model, mapper


def get_behavior_classifier(dim_rep):
    vec_input1 = Input(shape=(dim_rep,))
    vec_input2 = Input(shape=(512,))
    vec = Concatenate(axis=-1)([vec_input1,vec_input2])
    score = Dense(256,activation='relu')(vec)
    score = Dense(256,activation='relu')(score)
    score = Dense(1,)(score)

    return Model([vec_input1,vec_input2],score)

def get_user_info_classifier(dim_rep,mapper,gamma=0.1):
    
    targe_input = Input(shape=(dim_rep,))
    pos_input = Input(shape=(512,))
    neg_input = Input(shape=(512,))

    
    classifier = get_behavior_classifier(dim_rep)
    
    pos_score = classifier([targe_input,pos_input])
    neg_score = classifier([targe_input,neg_input])
    scores = Concatenate(axis=-1)([pos_score,neg_score]) #(2,)
    
    logits = Activation('softmax')(scores)
    

    cons_dis =  Model([targe_input,pos_input,neg_input],logits)
    cons_dis.compile(loss=['categorical_crossentropy',],
                  optimizer=Adam(lr=0.0001), 
                  metrics=['acc'])
    
    data_rep_input = Input(shape=(512,))
    protected_rep = mapper(data_rep_input)
    logits = cons_dis([protected_rep,pos_input,neg_input])
    cons_att =  Model([data_rep_input,pos_input,neg_input],logits)
    cons_att.compile(loss=['categorical_crossentropy',],
                  optimizer=Adam(lr=0.0001*gamma), 
                  metrics=['acc'])    
    
    return cons_dis, cons_att

def get_eval_classifier():
    inputx = Input(shape=(512,))
    v = Dense(256,)(inputx)
    v = Dense(128,activation='relu')(v)
    v = Dense(128,activation='relu')(v)
    v = Dense(128,activation='relu')(v)
    y = Dense(2,activation='softmax')(v)
    model =  Model(inputx,y)
    model.compile(loss=['categorical_crossentropy',],
                  optimizer=Adam(lr=0.0001), 
                  metrics=['acc'])
    return model


def get_data_flow(train_data,lr,hyper_gender,hyper_age,gamma=0.1):
    
    #fairness-insensitive feature platform 1
    image_input = Input(shape=train_data[0].shape[1:],)

    image_rep = Conv2D(32,(5,5),)(image_input)
    image_rep = MaxPool2D((2,2))(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    
    image_rep = Conv2D(128,(5,5),)(image_rep)
    image_rep = MaxPool2D((2,2))(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    
    image_rep = Flatten()(image_rep)
    image_rep = Dense(512,activation='relu')(image_rep)
    
    
    #fairness-insensitive feature platform 2
    feature_input = Input(shape=train_data[1].shape[1:])
    feature_rep = Dense(32,activation='relu')(feature_input)
    feature_rep = Dense(32,activation='relu')(feature_rep)
    feature_rep = Dense(32)(feature_rep)
    
    
    # aggergation model
    raw_data_rep = Concatenate()([image_rep,feature_rep])
    raw_data_rep = Dense(512)(raw_data_rep)
    
    #target task classification
    vec = Dropout(0.2)(raw_data_rep)
    vec = Dense(512,activation='relu')(vec)
    vec = Dropout(0.2)(vec)
    vec = Dense(512,activation='relu')(vec)
    vec = Dropout(0.2)(vec)
    logit = Dense(2,activation='softmax')(vec)
    
    # build attribute mapper and attribute discirmator 
    gender_model, gender_mapper = get_discrimator()
    age_model, age_mapper = get_discrimator()
    
    gender_model.trainable = False
    age_model.trainable = False
    
    gender_logit = gender_model(raw_data_rep)
    age_logit = age_model(raw_data_rep)
    
    
    input_features = [image_input,feature_input]
    
    model = Model(input_features,[logit,gender_logit,age_logit])
    model.compile(loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'],
                      optimizer= Adam(lr=lr),
                      loss_weights=[1,hyper_gender,hyper_age],
                      metrics=['acc'])
    
    rep_model = Model(input_features,raw_data_rep)
    
    #build contrastive adversarial learning model
    gender_cons_adv = get_user_info_classifier(128,gender_mapper,gamma)
    age_cons_adv = get_user_info_classifier(128,age_mapper,gamma)

    return model,rep_model,gender_model,age_model, gender_mapper, age_mapper, gender_cons_adv, age_cons_adv