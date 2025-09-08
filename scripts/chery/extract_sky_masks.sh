segformer_path=third_party/SegFormer-master

python datasets/tools/extract_masks.py \
    --data_root data/chery/processed/training \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --split_file data/chery_example_scenes.txt \
    --process_dynamic_mask