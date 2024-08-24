### VGG11 ###
#1. BINARY
python main.py \
    --task binary \
    --net VGG11 \
    --custom_afs \
    --activation ReLU \
    --output_dir ./output/Select/VGG11-bin-relu \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net VGG11 \
    --custom_afs \
    --activation LeakyReLU \
    --output_dir ./output/Select/VGG11-bin-lrelu \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net VGG11 \
    --custom_afs \
    --activation LessNegativeReLU_0.03 \
    --output_dir ./output/Select/VGG11-bin-lnrelu-3 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume
    
python main.py \
    --task binary \
    --net VGG11 \
    --custom_afs \
    --activation LessNegativeReLU_0.05 \
    --output_dir ./output/Select/VGG11-bin-lnrelu-5 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net VGG11 \
    --custom_afs \
    --activation LessNegativeReLU_0.07 \
    --output_dir ./output/Select/VGG11-bin-lnrelu-7 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net VGG11 \
    --custom_afs \
    --activation LessNegativeReLU_0.09 \
    --output_dir ./output/Select/VGG11-bin-lnrelu-9 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

#2. SUBTYPE
python main.py \
    --task subtype \
    --net VGG11 \
    --custom_afs \
    --activation ReLU \
    --output_dir ./output/Select/VGG11-sub-relu \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net VGG11 \
    --custom_afs \
    --activation LeakyReLU \
    --output_dir ./output/Select/VGG11-sub-lrelu \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net VGG11 \
    --custom_afs \
    --activation LessNegativeReLU_0.03 \
    --output_dir ./output/Select/VGG11-sub-lnrelu-3 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume
    
python main.py \
    --task subtype \
    --net VGG11 \
    --custom_afs \
    --activation LessNegativeReLU_0.05 \
    --output_dir ./output/Select/VGG11-sub-lnrelu-5 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net VGG11 \
    --custom_afs \
    --activation LessNegativeReLU_0.07 \
    --output_dir ./output/Select/VGG11-sub-lnrelu-7 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net VGG11 \
    --custom_afs \
    --activation LessNegativeReLU_0.09 \
    --output_dir ./output/Select/VGG11-sub-lnrelu-9 \
    --batch_size 32 \
    --lr 1e-4 \
    --resume