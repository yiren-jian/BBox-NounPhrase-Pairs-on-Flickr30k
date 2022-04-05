from os import listdir
from os.path import isfile, join
# import random
# import cv2

if __name__ == '__main__':
    data_path = 'method2/'
    filenames = [f for f in listdir(data_path) if isfile(join(data_path, f)) if f.endswith('png')]

    with open(data_path[:-1]+'.md', "w") as f:
        for filename in filenames:
            if filename.startswith('PD'):
                to_write_down = "<img src='%s%s' width='400'>"%(data_path, filename)
                f.write(to_write_down)
