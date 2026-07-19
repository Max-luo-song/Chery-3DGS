uv run ns-train splatfacto \
    --data dataset/d3dr_dataset/synthetic/office_2/obj \
    --output-dir outputs/raw_3dgs/office_2-obj \
    --pipeline.model.background-color black \
    --viewer.quit-on-train-completion True nerfstudio-data \
    --orientation-method none --center-method none --auto-scale-poses False

uv run ns-train splatfacto \
    --data dataset/d3dr_dataset/synthetic/office_2/scene_eval/ \
    --output-dir outputs/raw_3dgs/office_2-scene_eval \
    --pipeline.model.background-color black \
    --viewer.quit-on-train-completion True nerfstudio-data \
    --orientation-method none --center-method none --auto-scale-poses False

uv run python3 train_everything.py \
      --exp_name "exp_retry" \
      --root outputs/ \
      --dataset_root dataset/d3dr_dataset/synthetic \
      --scene_name office_2 \
      --gaussian_splatting_root outputs/raw_3dgs/ \
      --transforms_obj dataset/d3dr_dataset/synthetic/office_2/obj_scene_eval \
      --use_personalization_from outputs/exp_office_2_012 \
      --skip_steps 0 1