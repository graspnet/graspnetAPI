import cv2
import os
save_dir = 'save'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for filename in os.listdir('.'):
    if filename.endswith('.py'):
        continue
    print(filename)
    img = cv2.imread(filename)
    save_img = os.path.join(save_dir, filename)
    # print()
    r_img = cv2.resize(img, dsize = (0, 0), fx = 0.5, fy = 0.5)
    cv2.imwrite(save_img, r_img)