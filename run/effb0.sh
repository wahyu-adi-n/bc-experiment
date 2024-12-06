### EfficientNetB0 ###
python main.py \
    --task binary \
    --net EfficientNetB0 \
    --output_dir ./output/Adam/EfficientNetB0-bin-relu \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net EfficientNetB0 \
    --output_dir ./output/AdamW/EfficientNetB0-bin-relu \
    --batch_size 32 \
    --optimizer AdamW \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net EfficientNetB0 \
    --output_dir ./output/Adamax/EfficientNetB0-bin-relu \
    --batch_size 32 \
    --optimizer Adamax \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net EfficientNetB0 \
    --output_dir ./output/RMSProp/EfficientNetB0-bin-relu \
    --batch_size 32 \
    --optimizer RMSProp \
    --epoch 50 \
    --lr 1e-4 \
    --resume
