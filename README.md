# 17644-Deep-Learning-Model-Inspection

### This is the release branch of the 17644 Model Inspection Team 1

## <ins> YOLOv3 and YOLOv4 </ins>

The YOLO networks are implemented using Tensorflow 2. The source code is available at here
https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3

The models of both neural networks are trained on AWS EC2 P3.2xlarge instances with Nvidia A100 graphic card.

The trained models can be downloaded at here. _**These weights files are only available to Google Drive account with andrew email.**_

YOLO v4: https://drive.google.com/drive/folders/1CWRKxbusRi68W9Gir1rEd92kGCmIFEq2?usp=sharing

YOLO v3: https://drive.google.com/drive/folders/1BoszspQVtNqOgN9kNEoSU9qMciOjOTLY?usp=sharing



## <ins> Attribution Methods - Integrated Gradient </ins>
The integrated methods are based on a self-built classification neural networks that trained on the public kaggle mask detection dataset. 
The source code of the attribution method is based on:
1. https://www.trulens.org/
2. https://www.tensorflow.org/tutorials/interpretability/integrated_gradients

The trained model of the self-built neural network is accessible at here: 
Model after training by 200 epochs: https://drive.google.com/drive/folders/1fx5AHpfKxUtTXtDmu4w2S065fD12OIHn?usp=sharing

Model using L1 regularization and early stopping: https://drive.google.com/drive/folders/17zyHpz1TEGLtwZJQHcmCdrsEjGguPybw?usp=sharing




## <ins> Dataset </ins>
The training datasets of all the models we used in this project are:
1. For YOLOv3 and YOLO v4: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
2. For self-trained model: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset

Please put all images into ```./Final_YOLOv3_YOLOv4/model_data/mask/images ```

## <ins>Requirements</ins>
### Run this command to install requirements.txt
```
pip install -r requirements.txt
```
### Recommended requirements file
```
numpy>=1.18.2
scipy>=1.4.1
wget>=3.2
seaborn>=0.10.0
tensorflow==2.5
tensorflow-gpu==2.5
opencv-python
tqdm==4.43.0
pandas
awscli
urllib3
mss
```
## <ins> Training </ins> 
We trained our Yolov3 and Yolov4 model both on AWS EC2 P3.2xlarge instance with the Tesla A100 graphic card. Later attempts have found that some of the reasonably good hyperparameters for batch sizes of 16 begin with a learning rate of 1e-4 and a learning rate of 1e-6. The training was carried out on the both two networks by 200 epoch.
To customize your training and evaluation options, make adjustments to ```./Final_YOLOv3_YOLOv4/yolov3/configs.py ```
Your weights will be stored at ```./Final_YOLOv3_YOLOv4/checkpoints```
### Command
Please go to ```./Final_YOLOv3_YOLOv4```, and run
```
python train.py
```
## <ins> Quick Demo</ins>
You can run this command to see a demo after the model is trained
```
python detection_demo.py
```
## <ins> Evaluation </ins>
Run command
```
python3 evaluate_mAP.py
```
You can also change different iou threshold by changing the parameters of ```get_mAP()```, the default iou threshold is 0.5
You will see the results at ```./Final_YOLOv3_YOLOv4/mAP```
