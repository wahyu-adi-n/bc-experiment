python main.py \
    --task binary \
    --net DenseNet121 \
    --activation ReLU \
    --output_dir ./output/Select/Test-DenseNet121-bin-relu \
    --batch_size 4 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet121 \
    --activation ReLU \
    --output_dir ./output/Select/Test-DenseNet121-sub-relu \
    --batch_size 4 \
    --lr 1e-4 \
    --resume