# Style Transfer on Fashion Products

This is the implementation of the Neural Style algorithm by [Gatys et al.](https://arxiv.org/abs/1508.06576) and \
DeepObjStyle (case of m=n) by [Mastan and Raman](https://arxiv.org/abs/2012.06498) for the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset). 

### Requirements
For downloading the libraries via conda, try:
```
conda install python=3.9
conda install pytorch=1.10 torchvision=0.11 cudatoolkit=11.3 -c pytorch
conda install pandas matplotlib
conda install scikit-learn opencv -c conda-forge 
conda install seaborn scikit-image -c anaconda
```

### Running
First create a folder named ```results``` to store the outputs.  
For running the optimization, please set method to 'neural_style' or 'deepobjstyle', choose input images, and try:
```
python main.py --method='deepobjstyle' --style_image_id=0 --content_image_id=0 
```

### Hyperparameters
For setting the weights of the loss terms, please refer to the notation of the papers mentioned in the beginning and try 
the following for DeepObjStyle:
```
python main.py --method='deepobjstyle' --alpha11=1e4 --alpha12=5e6 -alpha13=1e-4 --alpha2=1e-2
```
And the following for Neural Style:
```
python main.py --method='neural_style'--alpha=1 --beta=5e6
```

### Data and Segmentation Preprocessing
To randomly sample images from the Fashion Product Images Dataset, download the dataset from [kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) and
add the style images to the folder 'images/raw'. For preprocessing the images, please try:
```
python main.py --method='deepobjstyle' --prepare_data --path_style_raw='images\raw' --path_content_raw='fashion-product-images-dataset/fashion-dataset/images'
```
For creating respective segmentation images, please try: 
```
python main.py --method='deepobjstyle' --prepare_segmentation --path_segmentation='images/segmentation'
```

### Acknowlegements
The code is based on [NEURAL TRANSFER USING PYTORCH](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) and,
[yagudin/PyTorch-ddp-photo-styletransfer](https://github.com/yagudin/PyTorch-deep-photo-styletransfer).