segformer_path=third_party/SegFormer-master

python datasets/tools/extract_masks.py \
    --data_root data/chery/processed/training \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --scene_ids=clip_1746581703000 \
    --process_dynamic_mask