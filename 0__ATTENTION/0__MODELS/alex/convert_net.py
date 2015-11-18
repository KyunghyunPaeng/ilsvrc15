import os
import sys
sys.path.append('/home/paeng/projects/1__LIB/caffe-paeng/python') # caffe dependency
import caffe

if __name__ == "__main__" :
	
	# original network
	deploy = './test_ip.prototxt'
	model  = './pretrained_model_ip.caffemodel'
	net = caffe.Net(deploy, model, caffe.TEST)

	# fully conv network
	deploy = './test.prototxt'
	net_full_conv = caffe.Net(deploy, model, caffe.TEST)
	
	params = ['fc6', 'fc7', 'fc8']
	fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
	for fc in params:
		print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)	

	params_full_conv = ['conv6', 'conv7', 'conv8']
	conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

	for conv in params_full_conv:
		print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

	# converted network (fully conv. net)
	for pr, pr_conv in zip(params, params_full_conv) :
		conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
		conv_params[pr_conv][1][...] = fc_params[pr][1]
	
	save_name = './pretrained_model.caffemodel'
	net_full_conv.save(save_name)
	print 'fully conv model is saved in {}'.format(save_name)

