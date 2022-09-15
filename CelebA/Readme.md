## The codes of FairVFL on the CelebA dataset. 

**raw\_data\_process.py:** Preprocess the raw data of CelebA. This file should be executed when using this repo for the first time. The raw data should be first stored like the following structure. Then the code should be executed via this command ```python  raw_data_process.py -p data_path```. 

```
data_path
│---list_attr_celeba.txt
│---list_eval_partition.txt
└───data
	└─── unziped images from img_align_celeba.zip (e.g., 000000.png ...)

```
**preprocessing.py:** Load and preprocess the raw data for the fair vertical model learning.

**evaluation.py:** Evaluate the accuracy and fairness of the VML model.

**models.py:** Build the data flow of the vertical federateed learning model.

**train.py:** Build the vertical training workflow of FairVFL

**FairVFL.ipynb:** The startup of our FairVFL method.

**FairVFL.json:** Save the results of FairVFL. Each line represents the result of a model, which is selected via the validation dataset. Besides, each line has four column, which represents the target task accuracy, target task F1, gender fairness-F1, and age fairness-F1.



**Remark**: To simulate the VFL setting, we assume that the raw image and a part of raw features are stored in two platforms, and restrict their interactions in the model.