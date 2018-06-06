id=$1
todo=$2

if [ $todo = 'train' ]
then
    CUDA_VISIBLE_DEVICES=$id python train.py
fi

if [ $todo = 'eval' ]
then
    CUDA_VISIBLE_DEVICES=$id python evaluate.py
fi
