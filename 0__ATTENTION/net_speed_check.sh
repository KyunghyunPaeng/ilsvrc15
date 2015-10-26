
CAFFE_PATH=/home/paeng/projects/1__LIB/caffe-paeng

$CAFFE_PATH/build/tools/caffe time \
  -gpu $1 \
  -model 0__MODELS/$2/$3/train.prototxt \
  -iterations 20



