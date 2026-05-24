
from templates.data import get_bed_files, generate_shards_from_index
from pathlib import Path

bed_file = "IHECRE00000994.7_18_ChromHMM.bed.gz"

get_bed_files(file_names=[bed_file])

generate_shards_from_index()

"""
accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --gpu_ids 0,1,2,3 \
    --mixed_precision bf16 \
    playground/finetune_enformer.py \
    --data_dir ../sample/binned_dataframe_enformer/train_shards \
    --val_data_dir ../sample/binned_dataframe_enformer/valid_shards \
    --batch_size 2 \
    --epochs 2 \
    --lr 5e-5 \
    --output_dir enformer_finetuned.pt
"""