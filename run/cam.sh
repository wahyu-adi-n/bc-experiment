CUDA_VISIBLE_DEVICES=0 python gradcam.py \
    --method gradcam \
    --task subtype \
    --net DenseNet201 \
    --output-dir output/Select/DenseNet201-sub-lnrelu-3 \
    --image-path dataset/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-10926/40X/SOB_M_DC-14-10926-40-011.png

CUDA_VISIBLE_DEVICES=0 python gradcam.py \
    --method gradcam++ \
    --task subtype \
    --net DenseNet201 \
    --output-dir output/Select/DenseNet201-sub-lnrelu-3 \
    --image-path dataset/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-10926/40X/SOB_M_DC-14-10926-40-011.png