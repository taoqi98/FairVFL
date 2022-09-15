import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from preprocessing import *
from models import *
from evaluation import *
from train import *
import click



@click.command()
@click.option('-m', '--mode',default='Dense')
@click.option('-p', '--data-root-path',default='/data/qit16/FairML')
@click.option('-f', '--hyper-gender', default=1000)
@click.option('-y', '--hyper-age', default=100)
@click.option('-g', '--gamma', default=0.1)
@click.option('-i', '--gpu', default=0)
@click.option('-d', '--dim-rep', default=64)

def main(mode,hyper_gender,hyper_age,gamma,dim_rep,data_root_path,gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    session = tf.Session(config=config)
 
    KTF.set_session(session)

    LabelWeights, FeatureDict, TrainFeatures, ValFeatures, TestFeatures, TrainAttri, ValAttri, TestAttri, TrainLabels, ValLabels, TestLabels = load_data(data_root_path)
    SavedResult = []
    for repeat in range(40):
        Result = []
        model,rep_model,Discrimator,Mapper,Contrastive_Discrimator,Contrastive_Attacker = get_data_flow(mode,FeatureDict,hyper_gender,hyper_age,gamma)
        for i in range(10):
            train_model(model,rep_model,Discrimator,Mapper,Contrastive_Discrimator,Contrastive_Attacker,LabelWeights,TrainFeatures,TrainAttri,TrainLabels)
            val_result = evaluate_val(model,rep_model,LabelWeights,TrainFeatures,TrainAttri,ValFeatures,ValLabels,ValAttri)
            test_result = evaluate_test(model,rep_model,LabelWeights,TestFeatures,TestLabels,TestAttri)
            Result.append([test_result,val_result,])
            print(Result[-1])
        privacy = evaluate_privacy(rep_model,Mapper,TestFeatures,FeatureDict)
        print(privacy)
        Result.append(privacy)
        SavedResult.append(Result)
        dump_result(Result,'FairVFLv2.json')
        print()

if __name__ == '__main__':
    main()

