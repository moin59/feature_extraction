import sys
import os
import cv2
import numpy as np
caffe_root = '/home/wh/caffe/python'
sys.path.insert(0, os.path.join(caffe_root, 'caffe'))
import caffe
caffe.set_mode_gpu()

####class 1 for stand & class 0 for sitting
cap = cv2.VideoCapture(0)

while cv2.waitKey(1) < 0:
    ret,input_image_file = cap.read()

    ##### feature extract by caffenet
    # Berkeley Vision caffe reference model can be download from
    #  https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet
    model_file = 'bvlc_reference_caffenet.caffemodel'
    deploy_prototxt = 'deploy.prototxt'

    convNet = caffe.net(deploy_prototxt, model_file, caffe.TEST)
    layer = 'fc6'
    if layer not in convNet.blobs:
        raise TypeError("Invalid layer name: " + layer)

    imagemean_file = 'ilsvrc_2012_mean.npy'

    transformer = caffe.io.Transformer({'data': convNet.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(imagemean_file).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255.0)

    convNet.blobs['data'].reshape(1,3,227,227)
    # img = caffe.io.load_image(input_image_file)
    img =np.copy(input_image_file)
    convNet.blobs['data'].data[...] = transformer.preprocess('data', img)

    output = convNet.forward()

    # generated key-points value save into a txt file
    with open("feature_stand.txt", 'a') as file:
        abstract = convNet.blobs[layer].data[0]
        new_list = []
        for item in abstract:
            new_list.append(float(item))
        file.write(str(new_list) + '\n'+'"Standing",')
        print(new_list)