## The codes of FairVFL on the CelebA dataset. 

**preprocessing.py:** Load and preprocess the raw data for the fair vertical model learning.

**evaluation.py:** Evaluate the accuracy and fairness of the VML model.

**models.py:** Build the data flow of the vertical federateed learning model.

**train.py:** Build the vertical training workflow of FairVFL

**FairVFL.ipynb:** The startup of our FairVFL method.

**FairVFL.json:** Save the results of FairVFL. Each line represents the result of a model, which is selected via the validation dataset. Besides, each line contains two dictionaries. The first dictionary contains the target task accuracy, target task F1, gender fairness-F1, and age fairness-F1. The second dictionary contains the privacy leakage evaluation result on each input feature. The file structure is shown below, where L() denotes the privacy leakage result of the representation.

```
[{'performance':[accuracy,f1],'gender':gender_f1,'age':age_f1}, 
{feature_id:[L(protected_representation),L(unprotected_representation), random_guess]} ]

```

**Remark**: To simulate the VFL setting, we assume that the raw image and a part of raw features are stored in two platforms, and restrict their interactions in the model. Moreover, in our work, we study the privacy protection problem during the whole VFL training stage. Thus, for the privacy protection evaluation, we evaluate the privacy leakage of the model in each training epoch and select the serious privacy leakage result as the privacy leakage of this model. The results show that inferring the private user information from the protected representation is close to the random guess, while it is possible to infer much user privacy from the unprotected results. This phenomenon shows that our proposed contrastive adversarial learning method can effectively protect user privacy.
