CUDA_VISIBLE_DEVICES=0 python train.py --T=16 -b=6 --num-classes=300 --workers=8 --epochs=100 --print-freq=300 --output-dir='.logs'\
									--warmup-epochs=0 --warmup-lr=0.0001 --opt=sgd --weight-decay=0.0005 --decay-epochs=20  --lr=0.001  --decay-rate=0.1 
