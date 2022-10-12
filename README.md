# Usage

## Prerequisites
- Python        3.8
- Pytorch       1.7.1
- torchvision   0.8.2

Step-by-step installation

```
conda create --name MY_PROJ -y python=3.8
conda activate MY_PROJ

# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

# this installs required packages
pip install -r requirements.txt
```

## Data Preparation
- Download The Cityscapes Dataset, the GTAV Dataset, and The SYNTHIA Dataset

Symlink the required dataset
```
ln -s /path_to_cityscapes_dataset datasets/cityscapes
ln -s /path_to_gtav_dataset datasets/gtav
ln -s /path_to_synthia_dataset datasets/synthia
```

The data folder should be structured as follows
```
├── datasets/
│   ├── cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── gtav/
|   |   ├── images/
|   |   ├── labels/
|   |   ├── gtav_label_info.p
│   └──	synthia
|   |   ├── RAND_CITYSCAPES/
|   |   ├── synthia_label_info.p
│   └──	
```

Generate the label static files for GTAV/SYNTHIA Datasets by running
```
python datasets/generate_gtav_label_info.py -d datasets/gtav -o datasets/gtav/
python datasets/generate_synthia_label_info.py -d datasets/synthia -o datasets/synthia/
```


Generate the images filename list.
```
python datasets/file_list_generator.py
```

