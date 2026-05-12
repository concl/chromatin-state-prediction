import pandas as pd
from pathlib import Path
from data import extract_long_sequences, read_bed_file, BED_FILES

def generate_shards():
    train_dir = Path("../sample/binned_dataframe/train_shards")
    val_dir = Path("../sample/binned_dataframe/val_shards")
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    val_chromosomes = ["chr8", "chr9"]
    
    # Process only the first BED file as an example to avoid OOM/Disk space issues
    # You would loop over all BED_FILES inside this loop in production
    print(f"Reading {BED_FILES[0]}...")
    df = read_bed_file(BED_FILES[0])
    
    print("Splitting train and val chromosomes...")
    val_df = df[df["chrom"].isin(val_chromosomes)]
    train_df = df[~df["chrom"].isin(val_chromosomes)]
    
    print("Extracting Validation Sequences (chr8, chr9)...")
    val_records = extract_long_sequences(val_df)
    if not val_records.empty:
        val_records.to_parquet(val_dir / "val_chr8_chr9.parquet")
        
    print("Extracting Train Sequences (Other)...")
    # For memory safety, we process train chromosomes one by one
    for chrom, group in train_df.groupby("chrom"):
        print(f"  Extracting train sequence for {chrom}...")
        chrom_records = extract_long_sequences(group)
        if not chrom_records.empty:
            chrom_records.to_parquet(train_dir / f"train_{chrom}.parquet")

if __name__ == "__main__":
    generate_shards()
