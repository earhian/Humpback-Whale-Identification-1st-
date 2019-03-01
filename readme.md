# Kaggle Humpback Whale Identification Challenge  1st place code


#### Dependencies
- python==3.6
- torch==0.4.1
- torchvision==0.2.1


## Solution
https://www.kaggle.com/c/humpback-whale-identification/discussion/82366



### Train
line 301 in train.py
step 1.
        freeze = False
        model_name = 'senet154'
        min_num_class = 10
        checkPoint_start = 0
        lr = 3e-4
        until train map5 >= 0.98

step 2.
        freeze = True
        model_name = 'senet154'
        min_num_class = 0
        checkPoint_start = best checkPoint of step 1
        lr = 3e-4

step 3.
        freeze = True
        model_name = 'senet154'
        min_num_class = 0
        checkPoint_start = best checkPoint of step 2
        lr = 3e-5

### Test
line 99 in test.py
       checkPoint_start = best checkPoint of step 3

