### DENSENET201 ###
python main.py \
    --task binary \
    --net DenseNet201 \
    --output_dir ./output/Adam/DenseNet201-bin-relu \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet201 \
    --output_dir ./output/AdamW/DenseNet201-bin-relu \
    --batch_size 32 \
    --optimizer AdamW \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet201 \
    --output_dir ./output/Adamax/DenseNet201-bin-relu \
    --batch_size 32 \
    --optimizer Adamax \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet201 \
    --output_dir ./output/RMSProp/DenseNet201-bin-relu \
    --batch_size 32 \
    --optimizer RMSProp \
    --epoch 50 \
    --lr 1e-4 \
    --resume