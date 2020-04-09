# HSE_ML_alphabet_recognition

[![Build Status](https://travis-ci.com/xLoSyAsHx/HSE_ML_alphabet_recognition.svg?branch=master)](https://travis-ci.com/xLoSyAsHx/HSE_ML_alphabet_recognition)


Neural Network model for English alphabet recognition. Deep learning engine - PyTorch


Table of contents
=================


   * [Head](#hse_ml_alphabet_recognition)
   * [Table of contents](#table-of-contents)
   * [Current network accuracy](#current-network-accuracy)
      * [English low letters](#english-low-letters---92.95%)
   * [How to launch training](#how-to-launch-training)
   * [Tensorboard](#tensorboard)
   * [MNIST19 preprocessed sample](#mnist1919-preprocessed-sample)
   * [Graphs&Metrics](#graphs&metrics)
      * [English low letters](#english-low-letters)


# Current network accuracy

## English low letters - 92.95%

- Mean AUC 0.85 +- 0.09 (1 std. dev.)
- Epoches 500
- Worst predicted classes: ( 'i', 'l' ), ( 'g', 'q' )
-      i: Recall 0.41, Precision 0.85, F1 0.55, AUC 0.68
       l: Recall 0.99, Precision 0.63, F1 0.77, AUC 0.66
       g: Recall 0.69, Precision 0.90, F1 0.78, AUC 0.79
       q: Recall 0.91, Precision 0.75, F1 0.82, AUC 0.72

Additional arguments: -test-limit 4000

# How to launch training
-

    # clone project
    cd HSE_ML_alphabet_recognition
    python -m alphabet_recogniser.train -root-dir ./data -data-type low_letters -batch-size 1000 -e 500 --use-preprocessed-data -t-images 80 -t-logdir ./runs/  -test-limit 4000 -m-save-path ./models/ -t-precision-bar-freq 2 -t-cm-freq 2 -t-roc-auc-freq 2 -m-save-period 2
    
    tensorboard --logdir=runs --samples_per_plugin images=250,text=250


# Tensorboard
Link to tensorboard runs folder: https://drive.google.com/drive/folders/1NXvIQlLt4xiJKGWvfR6yruZhDGgsOubo?usp=sharing


# MNIST19 preprocessed samples
![Samples](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/MNIST19_preprocessed_samples.png)


# Graphs&Metrics

##  English low letters
![TrainTest_Loss](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/TrainTest_Loss_e500.png)

![confusion_matrix](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/confusion_matrix_e500.png)

![ROC_AUC](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/ROC_AUC_500e.png)

![Recall_Precision_F1](https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition/blob/master/misc/images/Recall_Precision_F1_e500.PNG)
