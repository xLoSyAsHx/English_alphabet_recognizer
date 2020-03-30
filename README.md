# HSE_ML_alphabet_recognition

[![Build Status](https://travis-ci.com/xLoSyAsHx/HSE_ML_alphabet_recognition.svg?branch=master)](https://travis-ci.com/xLoSyAsHx/HSE_ML_alphabet_recognition)

**Table of Contents**

[TOCM]

[TOC]

Neural Network model for English alphabet recognition. Deep learning engine - PyTorch

## Current network accuracy
### English low letters
82.52% per 15 classes
-train-limit 2000 -test-limit 1000

## How to launch training
<
# clone project
cd HSE_ML_alphabet_recognition
python -m alphabet_recogniser.train -root-dir ./data  -data-type low_letters -batch-size 1000 -train-limit 2000 -test-limit 1000 -e 100 --use-preprocessed-data -classes {a,b,c,d,e,f,g,h} -t-images 80 --shuffle-test

tensorboard --logdir=runs
>


Link to tensorboard runs folder: https://drive.google.com/drive/folders/1WoD2z5Qg3KR-ASyUJnsuVkLzEGuTov0r?usp=sharing
