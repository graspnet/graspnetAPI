__author__ = 'mhgou'

from graspnetAPI.utils.eval_utils import load_dexnet_model
from tqdm import tqdm
import pickle
import os

##### Change the root to your path #####
graspnet_root = '/home/gmh/graspnet'

##### Do NOT change this folder name #####
dex_folder = 'dex_models'
if not os.path.exists(dex_folder):
    os.makedirs(dex_folder)

model_dir = os.path.join(graspnet_root, 'models')
for obj_id in tqdm(range(88), 'dump models'):
    dex_model = load_dexnet_model(os.path.join(model_dir, '%03d' % obj_id, 'textured'))
    with open(os.path.join(dex_folder, '%03d.pkl' % obj_id), 'wb') as f:
        pickle.dump(dex_model, f)
    
