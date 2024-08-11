### DENSENET121 ###
#1. BINARY
python main.py \
    --task binary \
    --net DenseNet121 \
    --activation ReLU \
    --output_dir ./output/Select/DenseNet121-bin-relu \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet121 \
    --activation LeakyReLU \
    --output_dir ./output/Select/DenseNet121-bin-lrelu \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet121 \
    --activation ParametricReLU_0.1 \
    --output_dir ./output/Select/DenseNet121-bin-prelu-1 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume
    
python main.py \
    --task binary \
    --net DenseNet121 \
    --activation ParametricReLU_0.2 \
    --output_dir ./output/Select/DenseNet121-bin-prelu-2 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet121 \
    --activation ParametricReLU_0.3 \
    --output_dir ./output/Select/DenseNet121-bin-prelu-3 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet121 \
    --activation ParametricReLU_0.4 \
    --output_dir ./output/Select/DenseNet121-bin-prelu-4 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet121 \
    --activation ParametricReLU_0.5 \
    --output_dir ./output/Select/DenseNet121-bin-prelu-5 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

#2. SUBTYPE
python main.py \
    --task subtype \
    --net DenseNet121 \
    --activation ReLU \
    --output_dir ./output/Select/DenseNet121-sub-relu \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet121 \
    --activation LeakyReLU \
    --output_dir ./output/Select/DenseNet121-sub-lrelu \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet121 \
    --activation ParametricReLU_0.1 \
    --output_dir ./output/Select/DenseNet121-sub-prelu-1 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume
    
python main.py \
    --task subtype \
    --net DenseNet121 \
    --activation ParametricReLU_0.2 \
    --output_dir ./output/Select/DenseNet121-sub-prelu-2 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet121 \
    --activation ParametricReLU_0.3 \
    --output_dir ./output/Select/DenseNet121-sub-prelu-3 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet121 \
    --activation ParametricReLU_0.4 \
    --output_dir ./output/Select/DenseNet121-sub-prelu-4 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet121 \
    --activation ParametricReLU_0.5 \
    --output_dir ./output/Select/DenseNet121-sub-prelu-5 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume