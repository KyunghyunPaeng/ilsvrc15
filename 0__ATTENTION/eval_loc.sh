
CAFFE_PATH=/home/paeng/projects/1__LIB/caffe-paeng

nohup $CAFFE_PATH/build/tools/caffe test \
  -gpu $1 \
  -model 0__MODELS/$2/$3/eval.prototxt \
  -weights 0__MODELS/$2/$3/ft_models/attention_imagenet_loc_15_iter_$4.caffemodel \
  -iterations 40238 > 0__MODELS/$2/$3/eval_$4.log &
