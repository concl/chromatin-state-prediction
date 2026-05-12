
# Chromatin state prediction research

## Contents

**playground**: Notebooks and scripts for exploratory work

**templates**: Shared code for training, data processing, and evaluation


## Setting up
Run the following command to download the human genome:

```
python templates.data.py
```


## Finetuning Enformer:
```
accelerate launch \
  --multi_gpu \
  --num_processes 3 \
  --gpu_ids 1,2,3 \
  --mixed_precision bf16 \
  playground/finetune_enformer.py \
  --data_dir ../sample/binned_dataframe/train_shards \
  --val_data_dir ../sample/binned_dataframe/val_shards \
  --batch_size 2 \
  --epochs 1 \
  --lr 5e-5 \
  --output_dir enformer_finetuned.pt
```


## Evaluating Enformer:
```
python playground/evaluate_enformer.py \
    --model_path playground/checkpoints/enformer_step_100.pt \
    --data_dir ../sample/binned_dataframe/val_shards \
    --batch_size 1 --print_every 50
```
