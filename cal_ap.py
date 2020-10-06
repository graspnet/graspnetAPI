import numpy as np
def cal_ap(npy_file):
    acc = np.load(npy_file)
    seen_acc = acc[:29]
    unseen_acc = acc[29:59]
    novel_acc = acc[59:]
    print('----\nseen:')
    cal_group_ap(seen_acc)
    print('----\nunseen:')
    cal_group_ap(unseen_acc)
    print('----\nnovel:')
    cal_group_ap(novel_acc)
    # print(acc.shape)

def cal_group_ap(acc):
    '''
    **Input:**
    - acc: np.array of shape (scene numbers, 256, 50, 6)
    '''
    ap = np.mean(acc[:,:,:,:])
    ap04 = np.mean(acc[:,:,:,1])
    ap08 = np.mean(acc[:,:,:,3])
    print('ap:{}, ap 0.4:{}, ap 0.8:{}'.format(ap, ap04, ap08))
    return ap, ap04, ap08

if __name__=='__main__':
    print('========= Without Pretraining =============')
    ap0 = cal_ap('acc_newdata.npy')
    print('========= With Pretraining =============')
    ap1 = cal_ap('acc_pretrained.npy')