
# Stacked_hourglass
Stacked_hourglass

## Intetion
* Check and Study Pose Estimation (In Jupyter notebook)

## Installation Process
* numpy
* matplotlib
* tqdm
* opencv
* pytorch
* os argparse
* sys
* scipy
* importlib
* h5py
* time

## Getting Started
* Associated Dataset => ref.py, dp.py
* Associated Transform => img.py, misc.py, group.py
* Associated model, Loss => Loss.py, posenet.py, pose.py
* I fixed Loss => Loss_re.py
* Associated evaluation, train, test => test.py, train.py

## Explanation ref.py, dp.py
* random Rotate image
* 1/2 probability, RGB => Fixed Random Color 


## Loss.py and Loss_re.py
* Invisivle keypoint => Loss Conver to 0
* Author`s Dataset No Invisible human pose estimation(keypoint Coordinate)

## Information Loss
* Author make Loss => Batch_size and Stage Loss => Mean

## train.py, test.py
* Training for model you must join the terminal(python train.py -e yourmodelname) => -e initialization
* python train.py -c yourmodelname => -c continue your model
* python test.py -c yourmodelname => -c continue your model and test(Reload from train.py)

## Layer Visualization
* Conv Filter  
![image](https://user-images.githubusercontent.com/59610723/128319373-d2e8dbd5-1009-4b46-bb32-e03b2b26afde.png)
  
* Residual Filter  
![image](https://user-images.githubusercontent.com/59610723/128319493-602d907c-5c37-4b49-9dd4-2f8bb46d52b0.png)
  
* Hourglass Filter  
Image Error in Code  
N=1  
![image](https://user-images.githubusercontent.com/59610723/128320181-91ab8c5d-2a51-4bdc-a25d-cdc0d6aeccba.png)
  
N=2  
![image](https://user-images.githubusercontent.com/59610723/128320221-0a46efda-b07f-42f6-92a5-b4ba18f097f3.png)

* Merge Filter  
![image](https://user-images.githubusercontent.com/59610723/128320406-3b3e083f-119e-4e41-aea3-185b5ba15c0e.png)
  
* PoseNet Filter  
![image](https://user-images.githubusercontent.com/59610723/128320449-78e63407-feee-438c-bfea-e4ac2484c126.png)
  
  
## Copyright / End User License
* To Check github (https://github.com/princeton-vl/pytorch_stacked_hourglass)

## Contact Information
* qazw5741@naver.com or YangChangHee2251@gmail.com
