
# imagenet cls validation (50,000 images)
# batch size : 50

CAFFE_PATH=/home/paeng/projects/1__LIB/caffe-paeng

nohup $CAFFE_PATH/build/tools/caffe test \
  -gpu $1 \
  -model test.prototxt \
  -weights pretrained_model_bn_infer.caffemodel \
  -iterations 1000 > test.log &

