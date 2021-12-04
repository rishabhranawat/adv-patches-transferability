# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os
# sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb

dn.set_gpu(0)
net = dn.load_net("/home/rdr2143/darknet/cfg/yolov2.cfg", "/home/rdr2143/darknet/yolov2.weights", 0)
meta = dn.load_meta("cfg/coco.data")

waymo_files = os.listdir('/home/rdr2143/waymo-adv-dataset/adv/patched/')

for each in waymo_files:
    r = dn.detect(net, meta, each)
    print(r)
    break

