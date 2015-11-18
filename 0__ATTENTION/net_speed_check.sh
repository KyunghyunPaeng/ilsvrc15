
CAFFE_PATH=/home/paeng/projects/1__LIB/caffe-paeng

$CAFFE_PATH/build/tools/caffe ftime \
  -gpu $1 \
  -model 0__MODELS/$2/$3/train_for_speed.prototxt \
  -weights 0__MODELS/$2/$3/ft_models/attention_imagenet_loc_15_iter_153824.caffemodel \
  -iterations 20



