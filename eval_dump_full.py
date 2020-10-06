from graspnetAPI import GraspNetEval
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
ge = GraspNetEval(root = '/home/minghao/graspnet', camera = 'kinect', split = 'test')
scene_accuracy_list = []

p = Pool(processes=1)


for scene_id in range(100,101):
    scene_accuracy_list.append(p.apply_async(ge.eval_scene, args=(scene_id,'kinect','/ssd1/minghao/dump_full_newdata_pretrained')))

p.close()
p.join()

accs = [scene_accuracy_list[i].get() for i in range(len(scene_accuracy_list))]
print(accs)
np.save('dump_full_newdata_pretrained_eval_result.npy', np.array(accs))
