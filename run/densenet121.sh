### DENSENET121 ###
python main.py \
    --task binary \
    --net DenseNet121 \
    --output_dir ./output/Adam/DenseNet121-bin-relu \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet121 \
    --output_dir ./output/AdamW/DenseNet121-bin-relu \
    --batch_size 32 \
    --optimizer AdamW \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet121 \
    --output_dir ./output/Adamax/DenseNet121-bin-relu \
    --batch_size 32 \
    --optimizer Adamax \
    --epoch 50 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet121 \
    --output_dir ./output/RMSProp/DenseNet121-bin-relu \
    --batch_size 32 \
    --optimizer RMSProp \
    --epoch 50 \
    --lr 1e-4 \
    --resume