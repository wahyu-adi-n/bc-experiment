### EfficientNetB7 ###
python main.py \
    --task binary \
    --net EfficientNetB7 \
    --output_dir ./output/Adam/EfficientNetB7-bin-relu \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net EfficientNetB7 \
    --output_dir ./output/AdamW/EfficientNetB7-bin-relu \
    --batch_size 32 \
    --optimizer AdamW \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net EfficientNetB7 \
    --output_dir ./output/Adamax/EfficientNetB7-bin-relu \
    --batch_size 32 \
    --optimizer Adamax \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net EfficientNetB7 \
    --output_dir ./output/RMSProp/EfficientNetB7-bin-relu \
    --batch_size 32 \
    --optimizer RMSProp \
    --epoch 50 \
    --lr 1e-4 \
    --resume
