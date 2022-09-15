import keras
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.models import Model

class Attention(Layer):
 
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
 
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
 
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x

        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))

        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)

        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def get_mapper(num,dim_rep):
    inputx = Input(shape=(256,))
    v = Dense(256,)(inputx)
    v = Dense(128,activation='relu')(v)
    v = Dense(128,activation='relu')(v)
    v = Dense(dim_rep,)(v)
    
    mapper = Model(inputx,v)
    
    return mapper

def get_mapper_discrimator(num,dim_rep):
    inputx = Input(shape=(256,))
    
    mapper = get_mapper(num,dim_rep)
    
    v = mapper(inputx)
    
    # discrimator
    y = Dense(num,activation='softmax')(v)
    model =  Model(inputx,y)
    model.compile(loss=['categorical_crossentropy',],
                  optimizer=Adam(lr=0.0001), 
                  #optimizer= SGD(lr=0.01),
                  metrics=['acc'])
    
    return model, mapper

def get_contrastive_scorer(dim_rep):
    vec_input1 = Input(shape=(dim_rep,))
    vec_input2 = Input(shape=(256,))
    vec = Concatenate(axis=-1)([vec_input1,vec_input2])
    score = Dense(256,activation='relu')(vec)
    score = Dense(256,activation='relu')(score)
    score = Dense(1,)(score)

    return Model([vec_input1,vec_input2],score)

def get_contrastive_discrimator(dim_rep):
    
    targe_input = Input(shape=(dim_rep,))
    pos_input = Input(shape=(256,))
    neg_input = Input(shape=(256,))

    
    scorer = get_contrastive_scorer(dim_rep)
    
    pos_score = scorer([targe_input,pos_input])
    neg_score = scorer([targe_input,neg_input])
    scores = Concatenate(axis=-1)([pos_score,neg_score]) #(2,)
    
    logits = Activation('softmax')(scores)
    

    model =  Model([targe_input,pos_input,neg_input],logits)
    model.compile(loss=['categorical_crossentropy',],
                  optimizer=Adam(lr=0.0001), 
                  metrics=['acc'])
    
    return model

def get_contrastive_attacker(gamma,mapper,contrastive_discrimator):
    target_data_input = Input(shape=(256,))
    pos_data_input = Input(shape=(256,))
    neg_data_input = Input(shape=(256,))
    protected_data_rep = mapper(target_data_input)
    
    pred = contrastive_discrimator([protected_data_rep,pos_data_input,neg_data_input])
    
    contrastive_attacker = Model([target_data_input,pos_data_input,neg_data_input],pred)
    
    contrastive_attacker.compile(loss=['categorical_crossentropy',],optimizer=Adam(lr=0.0001*gamma),metrics=['acc'])

    return contrastive_attacker

def get_fairness_attacker(num):
    inputx = Input(shape=(256,))
    v = Dense(256,)(inputx)
    v = Dense(128,activation='relu')(v)
    v = Dense(128,activation='relu')(v)
    v = Dense(128,activation='relu')(v)
    y = Dense(num,activation='softmax')(v)
    model =  Model(inputx,y)
    model.compile(loss=['categorical_crossentropy',],
                  optimizer=Adam(lr=0.0001), 
                  #optimizer= SGD(lr=0.01),
                  metrics=['acc'])
    return model


def get_local_model(mode):
    vec_inputs = [Input(shape=(32,)) for i in range(4)]
    rsp = Reshape((1,32))
    if mode == 'Dense':
        vecs = Concatenate(axis=-1)(vec_inputs)
        vec = Dense(32*4)(vecs)
    elif mode == 'Trans':
        vecs = [rsp(v) for v in vec_inputs]
        vecs = Concatenate(axis=-2)(vecs) #(4,32)
        vecs2 = Attention(2,16)([vecs]*3) #(4,32)
        vecs2 = Add()([vecs,vecs2])
        vec = Reshape((4*32,))(vecs)
        
    return Model(vec_inputs,vec)


def get_data_flow(mode,FeatureDict,hyper_gender=1/3,hyper_age=1/3,gamma=0.1):
    
    fnum = len(FeatureDict)
    dim_rep = 64
    feature_input = Input(shape=(fnum,),dtype='int32')
    feature_inputs = [Lambda(lambda x:x[:,i])(feature_input) for i in range(fnum)]
    
    FeatureEmbeddingLayers = [Embedding(FeatureDict[i].word_index, 32, trainable=True) for i in range(fnum)]
    
    feature_vecs = [FeatureEmbeddingLayers[i](feature_inputs[i]) for i in range(fnum)]
    feature_vecs1 = feature_vecs[:4]
    feature_vecs2 = feature_vecs[4:8]
    feature_vecs3 = feature_vecs[8:12]
    
    local_model1 = get_local_model(mode)
    local_model2 = get_local_model(mode)
    local_model3 = get_local_model(mode)

    feature_vec1 = local_model1(feature_vecs1)
    feature_vec2 = local_model2(feature_vecs2)
    feature_vec3 = local_model3(feature_vecs3)

    feature_vec = Concatenate(axis=-1)([feature_vec1,feature_vec2,feature_vec3])
    
    feature_vec = Dense(256,activation='relu')(feature_vec)
    user_vec = Dense(256,activation='relu')(feature_vec)

    scores = Dense(256,activation='relu')(user_vec)
    scores = Dense(256,activation='relu')(scores)
    logits = Dense(2,activation='softmax',name = 'target')(scores)

    Mapper = {}
    Discrimator = {}
    CateNums = {'gender':2,'age':4}
    
    Attack_logits = []
    
    for attr in ['gender','age']:
        discrimator,mapper = get_mapper_discrimator(CateNums[attr],dim_rep)
        
        Discrimator[attr] = discrimator
        Mapper[attr] = mapper
        
        discrimator.trainable = False
        attack_pred = discrimator(user_vec)
        attack_logit = Activation(keras.activations.softmax,name =attr)(attack_pred)  
        
        Attack_logits.append(attack_logit)
        
    model = Model(feature_input,[logits]+Attack_logits)
    model.compile(loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'],
                  optimizer=Adam(lr=0.0001), 
                  loss_weights=[1,hyper_gender,hyper_age],
                  metrics=['acc'])
    
    rep_model = Model(feature_input,user_vec)
    
    
    Contrastive_Discrimator = {}
    Contrastive_Attacker = {}
    for attr in Discrimator:
        Contrastive_Discrimator[attr] = get_contrastive_discrimator(dim_rep)
        Contrastive_Attacker[attr] = get_contrastive_attacker(gamma,Mapper[attr],Contrastive_Discrimator[attr])
    
    return model,rep_model,Discrimator,Mapper,Contrastive_Discrimator,Contrastive_Attacker