### RESNET152 ###
python main.py \
    --task binary \
    --net ResNet152 \
    --output_dir ./output/Adam/ResNet152-bin-relu \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net ResNet152 \
    --output_dir ./output/AdamW/ResNet152-bin-relu \
    --batch_size 32 \
    --optimizer AdamW \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net ResNet152 \
    --output_dir ./output/Adamax/ResNet152-bin-relu \
    --batch_size 32 \
    --optimizer Adamax \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net ResNet152 \
    --output_dir ./output/RMSProp/ResNet152-bin-relu \
    --batch_size 32 \
    --optimizer RMSProp \
    --epoch 50 \
    --lr 1e-4 \
    --resume