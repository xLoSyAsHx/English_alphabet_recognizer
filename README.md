# English alphabet recognizer

[![Build Status](https://travis-ci.com/xLoSyAsHx/English_alphabet_recognizer.svg?branch=master)](https://travis-ci.com/xLoSyAsHx/English_alphabet_recognizer)


Neural Network model for English alphabet recognition. Deep learning engine - PyTorch

Developed in the first year of the HSE university - Intellectual Data Analysis 


Table of contents
=================


   * [Current network accuracy](#current-network-accuracy)
   * [MNIST19 Dataset](#mnist19-dataset)
   * [Main requirements](#main-requirements)
   * [Installation](#installation)
   * [Train](#train)
   * [Test](#test)
   * [Demo](#demo)
   * [Best models](#best-models)
      * [Model EngAlphabetRecognizer96](#model-engalphabetrecognizer96)
        * [Snapshot](#snapshot)
   * [Tensorboard](#tensorboard)
   * [Graphs&Metrics](#graphs&metrics)
      * [EngAlphabetRecognizer96](#engalphabetrecognizer96)


# Current network accuracy

Dataset | Total accuracy | Mean AUC | Std. Dev. | Epoches | Train/Test Size |            | Worst predicted classes | F1 |
:--- | :---: | :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
MNIST19 | 92.95% | 0.85 | 0.09 | 500 | All / 3000  |            | ( 'i', 'l' ), ( 'g', 'q' ) | (0.55, 0.77), (0.78, 0.82) |


# MNIST19 Dataset

![Samples](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/MNIST19_preprocessed_samples.png)


# Main requirements

- torch - version 1.5.0
- torchvision - version 0.6.0
- tensorboardX - version 2.0
- scikit-learn - version 0.22.2.post1
- compress-pickle - version 1.1.1

# Installation

Alphabet recognizer is able to download, unzip and prepare dataset for use on its own. 

      git clone https://github.com/xLoSyAsHx/English_alphabet_recognizer.git
      cd English_alphabet_recognizer
      
      pip install -r requirements.txt
      
# Train

For train you need to specify config.cfg file

      cd English_alphabet_recognizer
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

      cd English_alphabet_recognizer
      python -m alphabet_recogniser.test -m ./path_to_model.model -root-dir ./data  -data-type low_letters -n 4000

Note: test.py works only with preprocessed-zipped images. Key '--use-preprocessed-data' for train.py

# Demo

      cd English_alphabet_recognizer
      python -m alphabet_recogniser.test -m ./models/2020_April04_15-55-56_acc[87.89]_e[150]_c[26]_tr_s[All]_t_s[2000].model -root-dir ./data  -data-type low_letters -n 4000

# Best models

## Model EngAlphabetRecognizer96

- Accuracy - 92.95%
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


# Graphs&Metrics

##  EngAlphabetRecognizer96
![TrainTest_Loss](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/TrainTest_Loss_e500.png)

![confusion_matrix](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/confusion_matrix_e500.png)

![ROC_AUC](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/ROC_AUC_500e.png)

![Recall_Precision_F1](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/Recall_Precision_F1_e500.PNG)
