segformer_path=third_party/SegFormer-master

scene_id="20250702_133223_Q2517"

python datasets/tools/extract_masks.py \
    --data_root data/qcraft/processed/training \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --scene_ids=$scene_id \
    --process_dynamic_mask