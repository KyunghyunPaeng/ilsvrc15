
CAFFE_PATH=/home/paeng/projects/1__LIB/caffe-paeng

nohup $CAFFE_PATH/build/tools/caffe compute_for_bn_inference \
  -gpu $1 \
  -model 0__MODELS/$2/$3/train_for_bn.prototxt \
  -weights 0__MODELS/$2/$3/ft_models/attention_voc_07_iter_39160.caffemodel \
  -iterations 15662 > 0__MODELS/$2/$3/compute_bn_infer_39160.log &



