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
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;            freeze = False  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;               model_name = 'senet154'  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;               min_num_class = 10  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             checkPoint_start = 0  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             lr = 3e-4  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             until train map5 >= 0.98  

step 2.  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             freeze = True  
 &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;            model_name = 'senet154'  
  &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;           min_num_class = 0  
    &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;         checkPoint_start = best checkPoint of step 1  
     &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;        lr = 3e-4  

step 3.  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             freeze = True  
 &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;       model_name = 'senet154'  
   &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;     min_num_class = 0  
   &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;     checkPoint_start = best checkPoint of step 2  
   &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;     lr = 3e-5  

### Test  
line 99 in test.py  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;       checkPoint_start = best checkPoint of step 3  

