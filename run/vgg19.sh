### VGG19 ###
python main.py \
    --task binary \
    --net VGG19 \
    --output_dir ./output/Adam/VGG19-bin-relu \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net VGG19 \
    --output_dir ./output/AdamW/VGG19-bin-relu \
    --batch_size 32 \
    --optimizer AdamW \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net VGG19 \
    --output_dir ./output/Adamax/VGG19-bin-relu \
    --batch_size 32 \
    --optimizer Adamax \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net VGG19 \
    --output_dir ./output/RMSProp/VGG19-bin-relu \
    --batch_size 32 \
    --optimizer RMSProp \
    --epoch 50 \
    --lr 1e-4 \
    --resume