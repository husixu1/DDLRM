# Multi

heturun -c config/config_local.yaml python3 train_resnet.py --batch-size 64 --model resnet50 --dataset cifar100 --learning-rate 0.01 --weight-decay 1e-4 --epochs 100 --pipeline pipedream --preduce
# Single
# env DMLC_PS_ROOT_URI=172.191.0.5 DMLC_PS_ROOT_PORT=13102 DMLC_NUM_WORKER=4 DMLC_NUM_SERVER=1 DMLC_PS_VAN_TYPE=p3 DMLC_ROLE=worker python3 train_resnet.py --batch-size 64 --model resnet50 --dataset cifar100 --learning-rate 0.01 --weight-decay 1e-4 --epochs 100 --pipeline pipedream --preduce
