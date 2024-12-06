### VGG11 ###
python main.py \
    --task binary \
    --net VGG11 \
    --output_dir ./output/Adam/VGG11-bin-relu \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net VGG11 \
    --output_dir ./output/AdamW/VGG11-bin-relu \
    --batch_size 32 \
    --optimizer AdamW \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net VGG11 \
    --output_dir ./output/Adamax/VGG11-bin-relu \
    --batch_size 32 \
    --optimizer Adamax \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net VGG11 \
    --output_dir ./output/RMSProp/VGG11-bin-relu \
    --batch_size 32 \
    --optimizer RMSProp \
    --epoch 50 \
    --lr 1e-4 \
    --resume