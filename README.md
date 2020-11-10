# graspnetAPI

GraspNet API comming soon!

## Dataset

Visit the [GraspNet Website](http://graspnet.net) to get the dataset.

## Document

Refer to [document](http://graspnet.net/docs/index.html) for more details.(Not availabel now)

You can also build the doc manually.
```bash
cd docs
pip install -r requirements.txt
bash build_doc.sh
```

LaTeX is required to build the pdf, but html can be built anyway.

## Install

```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
bash install.sh
```

## Examples
```bash
cd examples

# change the path of graspnet root

# How to load labels from graspnet.
python3 exam_loadGrasp.py

# How to convert between 6d and rectangle grasps.
python3 exam_convert.py

# Check the completeness of the data.
python3 exam_check_data.py

# you can also run other examples
```

<!-- ## Rectangle Grasp Labels

1. Download the rectangle label file [rect_labels.tar.gz](https://graspnet.net/datasets.html).  
2. Unzip the file by running  
```bash
tar -xzvf rect_labels.tar.gz
```
3. Modify the root path for graspnet in copy_rect_labels.py.  
3. Run copy_rect_labels.py to copy the labels into the corresponding folder.
```bash
python3 copy_rect_labels.py
``` -->
