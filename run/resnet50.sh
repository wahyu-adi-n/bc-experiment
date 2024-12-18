### RESNET50 ###
python main.py \
    --task binary \
    --net ResNet50 \
    --output_dir ./output/Adam/ResNet50-bin-relu \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net ResNet50 \
    --output_dir ./output/AdamW/ResNet50-bin-relu \
    --batch_size 32 \
    --optimizer AdamW \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net ResNet50 \
    --output_dir ./output/Adamax/ResNet50-bin-relu \
    --batch_size 32 \
    --optimizer Adamax \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net ResNet50 \
    --output_dir ./output/RMSProp/ResNet50-bin-relu \
    --batch_size 32 \
    --optimizer RMSProp \
    --epoch 50 \
    --lr 1e-4 \
    --resume