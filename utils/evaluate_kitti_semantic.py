import numpy as np
import argparse
import os
import cv2
import tensorflow as tf

from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (2550,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (1255, 11, 32) ),
]


id2trainId = { label.id : label.trainId for label in labels }
ignore_label = 255
num_classes = 19

def convert_to_train_id(sem,id2trainId=id2trainId):
    p = tf.cast(sem,tf.uint8)
    m = tf.zeros_like(p)
    for i in range(0, len(labels)):
        mi = tf.multiply(tf.ones_like(p), id2trainId[i])
        m = tf.where(tf.equal(p,i), mi, m)
    return m

parser = argparse.ArgumentParser(description='Evaluation on the cityscapes validation set')
parser.add_argument('--pred_path',  type=str,   help='file to predictions semantic maps npy',   required=True)
parser.add_argument('--filenames_file', type=str,  help='file to filelist.txt', required=True)
parser.add_argument('--data_path',  type=str,   help='file to kitti dataset',   required=True)

args = parser.parse_args()

### INPUTS ###
sem_seg_placeholder = tf.placeholder(tf.int32)
sem_seg_placeholder.set_shape([None,None,1])
sem_gt_placeholder = tf.placeholder(tf.int32)
im_shape_w = tf.placeholder(tf.int32)
im_shape_h = tf.placeholder(tf.int32)

### RESIZE PREDICTIONS ###
sem_pred = tf.image.resize_images(sem_seg_placeholder, [im_shape_h, im_shape_w] ,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

### CONVERT TO IGNORE LABELS IN EVAL ###
sem_pred = convert_to_train_id(sem_pred)
sem_gt = convert_to_train_id(sem_gt_placeholder)

### INIT WEIGHTS MIOU
weightsValue = tf.to_float(tf.not_equal(sem_gt,ignore_label))

### IGNORE LABELS TO 0, WE HAVE ALREADY MASKED THOSE PIXELS WITH WEIGHTS 0###
sem_gt = tf.where(tf.equal(sem_gt, ignore_label), tf.zeros_like(sem_gt), sem_gt)
sem_pred = tf.where(tf.equal(sem_pred, ignore_label), tf.zeros_like(sem_pred), sem_pred)
### ACCURACY ###
acc, update_op_acc = tf.metrics.accuracy(sem_gt,sem_pred,weights=weightsValue)

### MIOU ###
miou, update_op = tf.metrics.mean_iou(labels=tf.reshape(sem_gt,[-1]),predictions=tf.reshape(sem_pred,[-1]), num_classes=num_classes, weights=tf.reshape(weightsValue,[-1]))

### INIT OP ###
init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

miou_value = 0
with tf.Session() as sess:
    sess.run(init_op)
    pred_segs= np.load(args.pred_path)
    with open(args.filenames_file) as filelist:
        for idx,line in enumerate(filelist):
            print(args.data_path)
            fpath = args.data_path + line.split(" ")[-1].strip()
            semgt = cv2.imread(fpath,cv2.IMREAD_GRAYSCALE)
            sem = pred_segs[idx]
            print(fpath)  

            acc_value, _, miou_value, _ =sess.run([acc, update_op_acc, miou, update_op],feed_dict={sem_seg_placeholder : np.expand_dims(sem,axis=-1) , sem_gt_placeholder : semgt, im_shape_w: semgt.shape[1], im_shape_h: semgt.shape[0] })
                     

print "mIoU: " , miou_value, "acc", acc_value
