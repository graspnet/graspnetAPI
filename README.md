# graspnetAPI

GraspNet API comming soon!

## Install

```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
bash install.sh
```

## Example
```bash
cd examples

# change the path of graspnet root
python3 exam_loadGrasp.py

# you can also run other examples

```

## Rectangle Grasp Labels

1. Download the rectangle label file [rect_labels.tar.gz](https://graspnet.net/datasets.html).  
2. Unzip the file by running  
```bash
tar -xzvf rect_labels.tar.gz
```
3. Modify the root path for graspnet in copy_rect_labels.py.  
3. Run copy_rect_labels.py to copy the labels into the corresponding folder.
```bash
python3 copy_rect_labels.py
```
