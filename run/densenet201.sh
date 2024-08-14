### DENSENET201 ###
#1. BINARY
python main.py \
    --task binary \
    --net DenseNet201 \
    --activation ReLU \
    --output_dir ./output/Select/DenseNet201-bin-relu \
    --batch_size 64 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet201 \
    --activation LeakyReLU \
    --output_dir ./output/Select/DenseNet201-bin-lrelu \
    --batch_size 64 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet201 \
    --activation LessNegativeReLU_0.03 \
    --output_dir ./output/Select/DenseNet201-bin-lnrelu-3 \
    --batch_size 64 \
    --lr 1e-4 \
    --resume
    
python main.py \
    --task binary \
    --net DenseNet201 \
    --activation LessNegativeReLU_0.05 \
    --output_dir ./output/Select/DenseNet201-bin-lnrelu-5 \
    --batch_size 64 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet201 \
    --activation LessNegativeReLU_0.07 \
    --output_dir ./output/Select/DenseNet201-bin-lnrelu-7 \
    --batch_size 64 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet201 \
    --activation LessNegativeReLU_0.09 \
    --output_dir ./output/Select/DenseNet201-bin-lnrelu-9 \
    --batch_size 64 \
    --lr 1e-4 \
    --resume

#2. SUBTYPE
python main.py \
    --task subtype \
    --net DenseNet201 \
    --activation ReLU \
    --output_dir ./output/Select/DenseNet201-sub-relu \
    --batch_size 64 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet201 \
    --activation LeakyReLU \
    --output_dir ./output/Select/DenseNet201-sub-lrelu \
    --batch_size 64 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet201 \
    --activation LessNegativeReLU_0.03 \
    --output_dir ./output/Select/DenseNet201-sub-lnrelu-3 \
    --batch_size 64 \
    --lr 1e-4 \
    --resume
    
python main.py \
    --task subtype \
    --net DenseNet201 \
    --activation LessNegativeReLU_0.05 \
    --output_dir ./output/Select/DenseNet201-sub-lnrelu-5 \
    --batch_size 64 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet201 \
    --activation LessNegativeReLU_0.07 \
    --output_dir ./output/Select/DenseNet201-sub-lnrelu-7 \
    --batch_size 64 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet201 \
    --activation LessNegativeReLU_0.09 \
    --output_dir ./output/Select/DenseNet201-sub-lnrelu-9 \
    --batch_size 64 \
    --lr 1e-4 \
    --resume