import os
import shutil
import numpy as np

attr = np.load('celeba_attr.npz')['attributes']

splits = [('train', [1,799]), ('val', [800,1000])]
classes = ['female', 'male']
male_attr_idx = 20
for s in splits:
    os.mkdir(s[0])
    classes_path = [os.path.join(s[0],name) for name in classes]
    for p in classes_path:
        os.mkdir(p)
    for i in range(s[1][0],s[1][1]):
        filename = format(i, '0>6,d')+'.jpg'
        src = os.path.join('imgs',filename)
        dst = os.path.join(classes_path[attr[i-1][male_attr_idx]], filename)
        shutil.copy(src, dst)
        