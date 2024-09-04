### DENSENET121 ###
#1. BINARY
python main.py \
    --task binary \
    --net DenseNet121 \
    --custom_afs \
    --activation ReLU \
    --output_dir ./output/Select/DenseNet121-bin-relu \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 20 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet121 \
    --custom_afs \
    --activation LeakyReLU \
    --output_dir ./output/Select/DenseNet121-bin-lrelu \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 20 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet121 \
    --custom_afs \
    --activation LessNegativeReLU_0.03 \
    --output_dir ./output/Select/DenseNet121-bin-lnrelu-3 \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 20 \
    --lr 1e-4 \
    --resume
    
python main.py \
    --task binary \
    --net DenseNet121 \
    --custom_afs \
    --activation LessNegativeReLU_0.05 \
    --output_dir ./output/Select/DenseNet121-bin-lnrelu-5 \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 20 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet121 \
    --custom_afs \
    --activation LessNegativeReLU_0.07 \
    --output_dir ./output/Select/DenseNet121-bin-lnrelu-7 \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 20 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet121 \
    --custom_afs \
    --activation LessNegativeReLU_0.09 \
    --output_dir ./output/Select/DenseNet121-bin-lnrelu-9 \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 20 \
    --lr 1e-4 \
    --resume

#2. SUBTYPE
python main.py \
    --task subtype \
    --net DenseNet121 \
    --custom_afs \
    --activation ReLU \
    --output_dir ./output/Select/DenseNet121-sub-relu \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 20 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet121 \
    --custom_afs \
    --activation LeakyReLU \
    --output_dir ./output/Select/DenseNet121-sub-lrelu \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 20 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet121 \
    --custom_afs \
    --activation LessNegativeReLU_0.03 \
    --output_dir ./output/Select/DenseNet121-sub-lnrelu-3 \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 20 \
    --lr 1e-4 \
    --resume
    
python main.py \
    --task subtype \
    --net DenseNet121 \
    --custom_afs \
    --activation LessNegativeReLU_0.05 \
    --output_dir ./output/Select/DenseNet121-sub-lnrelu-5 \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 20 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet121 \
    --custom_afs \
    --activation LessNegativeReLU_0.07 \
    --output_dir ./output/Select/DenseNet121-sub-lnrelu-7 \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 20 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet121 \
    --custom_afs \
    --activation LessNegativeReLU_0.09 \
    --output_dir ./output/Select/DenseNet121-sub-lnrelu-9 \
    --batch_size 32 \
    --optimizer Adam \
    --epoch 20 \
    --lr 1e-4 \
    --resume