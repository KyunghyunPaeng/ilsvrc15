
CAFFE_PATH=/home/paeng/projects/1__LIB/caffe-paeng

nohup $CAFFE_PATH/build/tools/caffe compute_for_bn_inference \
  -gpu $1 \
  -model train.prototxt \
  -weights pretrained_model.caffemodel \
  -iterations 40000 > compute_bn_infer.log &

