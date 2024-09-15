# Feature Remover

This repository was created as part of my Bachelor's thesis "Removing Texture Features using Generative Adversarial Networks". 

# Background
In my thesis, I worked on the GRASS image processing project at Technische Hochschule Würzburg Schweinfurt, in which ways to automate intralogistic processes are researched. 

This project uses a box recognition algorithm for an efficient de-palletization process. 
One issue herein is several prints, stickers, and alike (herein called features) that impede the box recognition algorithm.
My task was to train and evaluate GANs that remove such features and therefore facilitate efficient recognition. 

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
Now the scripts can be run:

### compare_models.py
This script infers images placed in the ./input folder using the pix2pix, CycleGAN, and Vit-GAN models to evaluate the various GAN models. 
The result of every image is displayed afterward. 
If wished, the images can be saved in the ./output folder by specifying the `--save` flag.

Additionally, an edge map can be created using the canny algorithm. To do so, the `--canny` flag has to be specified.

So to save the resulting images with a canny edge map, the following command shall be executed:
```bash
python3 compare_models.py --save True --canny True
```
