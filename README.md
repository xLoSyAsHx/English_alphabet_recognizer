# English alphabet recogniser

[![Build Status](https://travis-ci.com/xLoSyAsHx/HSE_ML_alphabet_recognition.svg?branch=master)](https://travis-ci.com/xLoSyAsHx/HSE_ML_alphabet_recognition)


Neural Network model for English alphabet recognition. Deep learning engine - PyTorch

Developed in the first year of the HSE university - Intellectual Data Analysis 


Table of contents
=================


   * [Current network accuracy](#current-network-accuracy)
   * [Requirements](#requirements)
   * [Instalation](#instalation)
   * [Train](#train)
   * [Test](#test)
   * [Demo](#demo)
   * [Best models](#best-models)
      * [English low letters](#english-low-letters--92.95%)
        * [Snapshot](#snapshot)
   * [Tensorboard](#tensorboard)
   * [Dataset](#dataset)
     * [MNIST19 preprocessed sample](#mnist1919-preprocessed-sample)
     * [Graphs&Metrics](#graphs&metrics)
        * [English low letters](#english-low-letters)


# Current network accuracy

Dataset | Total accuracy | Mean AUC | Std. Dev. | Epoches | Train/Test Size |            | Worst predicted classes | F1 |
:--- | :---: | :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
MNIST19 | 92.95% | 0.85 | 0.09 | 500 | All / 3000  |            | ( 'i', 'l' ), ( 'g', 'q' ) | (0.55, 0.77), (0.78, 0.82) |


# Requirements

      compress-pickle==1.1.1
      imageio==2.8.0
      matplotlib==3.2.1
      numpy==1.18.2
      opencv-python==4.2.0.34
      Pillow==7.0.0
      requests==2.23.0
      scikit-image==0.16.2
      scikit-learn==0.22.2.post1
      tensorboardX==2.0
      tensorflow==2.2.0rc3
      termcolor==1.1.0
      torch==1.5.0
      torchvision==0.6.0
      tqdm==4.43.0

# Instalation

      git clone https://github.com/xLoSyAsHx/English_alphabet_recogniser.git
      cd English_alphabet_recogniser
      
      pip install -r requirements.txt
      
# Train

For train you need to specify config.cfg file

      cd English_alphabet_recogniser
      python -m alphabet_recogniser.train -config ./data/config.cfg

Config.cgf:

      ################################################
      ### Example Config.cfg
      ################################################
      
      -root-dir ./data
      
      -data-type low_letters
      -batch-size 1000
      -e 4
      -classes {a,d,f} # For all classes, just comment this line  
      -train-limit 500
      -test-limit 50

      --use-preprocessed-data

      # Tensorboard settings
      -t-images 80
      -t-logdir ./runs/
      -t-precision-bar-freq 2
      -t-cm-freq 2
      -t-roc-auc-freq 2

      -m-save-path ./models/
      -m-save-period 2

# Test

      cd English_alphabet_recogniser
      python -m alphabet_recogniser.test -m ./path_to_model.model -root-dir ./data  -data-type low_letters -n 4000

Note: test.py works only with preprocessed-zipped images. Key '--use-preprocessed-data' for train.py

# Demo

      cd English_alphabet_recogniser
      python -m alphabet_recogniser.test -m ./models/2020_April04_15-55-56_acc[87.89]_e[150]_c[26]_tr_s[All]_t_s[2000].model -root-dir ./data  -data-type low_letters -n 4000

# Best models

## English low letters - 92.95%

- Mean AUC 0.85 +- 0.09 (1 std. dev.)
- Epoches 500
- Worst predicted classes: ( 'i', 'l' ), ( 'g', 'q' )
 
       i: Recall 0.41, Precision 0.85, F1 0.55, AUC 0.68
       l: Recall 0.99, Precision 0.63, F1 0.77, AUC 0.66
       g: Recall 0.69, Precision 0.90, F1 0.78, AUC 0.79
       q: Recall 0.91, Precision 0.75, F1 0.82, AUC 0.72

Additional arguments: -test-limit 4000

### Snapshot

Link to tensorboard runs folder: https://drive.google.com/drive/folders/18jtrvTQUPmu6me_AH7kdXUOeInkRYN3Q?usp=sharing

Link to models folder: https://drive.google.com/drive/folders/1hzrlxoFrR1zDp_1YM9eIk6m-nge-SwXW?usp=sharing


# Dataset

## MNIST19 preprocessed samples
![Samples](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/MNIST19_preprocessed_samples.png)


## Graphs&Metrics

###  English low letters
![TrainTest_Loss](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/TrainTest_Loss_e500.png)

![confusion_matrix](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/confusion_matrix_e500.png)

![ROC_AUC](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/ROC_AUC_500e.png)

![Recall_Precision_F1](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/Recall_Precision_F1_e500.PNG)
