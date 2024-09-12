# Feature Remover

This repository was created as part of my Bachelors' thesis "Removing Texture Features using Generative Adversarial Networks". 

# Background
In my thesis I worked on a the GRASS image processing project at Technische Hochschule Würzburg Schweinfurt, in which ways to automate intralogistic processes are researched. 

In this project a box recognition algorithm is used for an efficient de-palletization process. 
One issue herein are several prints, stickers and alike (herein called features) which impede the box recognition algorithm.
My task was to train and evaluate GANs which remove such features and therefore facilitate an efficient recognition. 

# How to run the software? 

## Clone this repository
```bash
git clone https://github.com/jaba17/feature_remover.git
```
## Install requirements
After creating a new virtual environment, install the requirements using pip:

```bash
pip install -r requirements.txt
```

## Place images in the appropriate folder
 
The input images shall be placed in the ./input folder. 


## Run the scripts
Now the programms can be run as 

### compare_models.py
In order to evaluate the various GAN models, this script infers images which are placed in the ./input folder using the pix2pix, CycleGAN and Vit-GAN model. 
The result of every image is displayed afterwards. 
If wished, the images can be saved in the ./output folder by specifying the --save flag.

Additionally, an edge map can be created using the canny algorithm. To do so, the `--canny` flag has to be specified.


### analyze_video.py
This script analyzes a video file which is g