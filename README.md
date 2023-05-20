# Deep Representation One-class Classification (DROC).
The present work is implementation of DROC on FathomNet Dataset for one-class classification of Marine objects. It is based on the work of 
Kihyuk Sohn, Chun-Liang Li, Jinsung Yoon, Minho Jin, and Tomas Pfister. 
Please refer below paper for more information.
[Learning and Evaluating Representations for Deep One-class Classification](https://openreview.net/forum?id=HCSgyPUfeDj) 
published at [ICLR 2021](https://iclr.cc/) as a conference paper
by Kihyuk Sohn, Chun-Liang Li, Jinsung Yoon, Minho Jin, and Tomas Pfister.

This directory has deep one-class classification which includes self-supervised deep representation learning from
one-class data, and a classifier using discriminative model OC-SVM. It also contains Baseline model based on state of the art 
OC-SVM.

## Install

The `requirements.txt` includes all the dependencies for this project.


## Download datasets

`Dataset/` includes an instruction how to download and prepare data for
FathomNet dataset. The dataset needs to be downloaded using the scripts prepare_data001, prepare_data002,... given in Dataset folder. 
Please download dataset prior to model training.

## Run

The options for the experiments are specified through the command line arguments.
The detailed explanation can be found in `train_and_eval_loop.py`. Scripts for
running experiments can be found

-   Contrastive learning with distribution augmentation in home directory:
    `run_contrastive_da.sh` (paths should be set prior in the code)
    
-   To run baseline:
    `run_Baseline.sh` (paths should be set prior in the code)

-   Other details for code are in source folder (Network architecture, util, etc.)
## Evaluation

After running `train_and_eval_loop.py` using `run_contrastive_da.sh`, the evaluation results can be found in
`$Output/.../stats/summary.json`, where `...` is specified as classwise folder.

-   For model prediction:
    `Model_presiction.sh` (paths should be updated in the file prior) 

-   Model can be found in saved_model folder in home directory
 
