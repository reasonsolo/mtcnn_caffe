# mtcnn_caffe

## dataset
- [WIDER](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
- [LFW](http://mmlab.ie.cuhk.edu.hk/archive/CNN\_FacePoint.htm)

Extract both training dataset to directory WIDER\_train and lfw\_train. Place annotation file/folder in each dataset's directory

## prepare data for pnet
0. Get enough free disk space.
1. Run `python gen_net_data.py pnet train` to generate raw input data for pnet phase.
2. Run `python gen_lmdb.py pnet train` to generate lmdb input for caffe.

## run training for pnet
Run `python solver_test.py pnet_solver.prototxt 500000`. Set last parameter to your preferred max iterating num.

## prepare data for rnet/onet
0. Get more enough free disk space.
1. Run `python gen_hard_example.py rnet train 500000`. Replace `rnet` with `onet` if necessary
and `500000` is your preferred caffemodel checkpoint in last training round.

## test pnet
Run `python test_net.py pnet 500000 PATH_TO_TEST_IMAGE`
