import numpy as np
import argparse
import os
import cv2
import subprocess

parser = argparse.ArgumentParser(description='Evaluation on the cityscapes validation set')
parser.add_argument('--cityscapes_path', type=str,   help='path to main folder of cityscapes', required=True)
parser.add_argument('--pred_path',  type=str,   help='file to predictions semantic maps',   required=True)
parser.add_argument('--filelist', type=str,  help='file to filelist.txt', required=True)
args = parser.parse_args()

pred_segs= np.load(args.pred_path)
outputPath = os.path.dirname(args.pred_path)
filelist = open(args.filelist)
print('Resizing '+ str(len(pred_segs)) +' images')
for idx,line in enumerate(filelist):
    fpath = line.split(" ")[0]
    sem = pred_segs[idx]
    sem=cv2.resize(sem,(2048,1024))
    cv2.imwrite(os.path.join(outputPath,fpath),sem)

os.environ['CITYSCAPES_DATASET'] = args.cityscapes_path
os.environ['CITYSCAPES_RESULTS'] = outputPath
p=subprocess.Popen("cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py",shell=True)  

def kill_child():
    if p.pid is None:
        pass
    else:
        p.kill()

import atexit
atexit.register(kill_child)

p.wait()  


