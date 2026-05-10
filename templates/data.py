
import os
import subprocess
import sys
import gzip
import pandas as pd

from pathlib import Path

PATH = Path(__file__).resolve().parent.parent
DOWNLOAD_PATH = PATH / "sample" / "human_genome"
BED_PATH = PATH / "sample" / "bed_files"
CHROMOSOMES = [
    f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]
]

BED_FILES = [
    "IHECRE00001475.1_18_ChromHMM.bed.gz",
    "IHECRE00002659.1_18_ChromHMM.bed.gz",
    "IHECRE00002660.1_18_ChromHMM.bed.gz",
    "IHECRE00004909.1_18_ChromHMM.bed.gz",
    "IHECRE00004910.1_18_ChromHMM.bed.gz",
]

SOURCE = "ftp://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/"

def get_all_chromosomes():
    for chrom in CHROMOSOMES:
        url = f"{SOURCE}{chrom}.fa.gz"
        output_path = DOWNLOAD_PATH / f"{chrom}.fa.gz"
        if not output_path.exists():
            print(f"Downloading {chrom}...")
            subprocess.run(["curl", "-o", str(output_path), url], check=True)
        else:
            print(f"{chrom} already exists, skipping download.")

def decompress_chromosome(chrom: str) -> str:
    gz_path = DOWNLOAD_PATH / f"{chrom}.fa.gz"
    if not gz_path.exists():
        raise FileNotFoundError(f"{gz_path} does not exist.")
    
    with gzip.open(gz_path, 'rt') as f:
        return f.read()

def read_bed_file(bed_file: str):
    bed_path = BED_PATH / bed_file
    if not bed_path.exists():
        raise FileNotFoundError(f"{bed_path} does not exist.")
    
    with gzip.open(bed_path, 'rt') as f:
        data = pd.read_csv(f, sep='\t', header=None)
        data.columns = ['chrom', 'start', 'end', 'state']
    return data

def main():
    if not DOWNLOAD_PATH.exists():
        DOWNLOAD_PATH.mkdir(parents=True)
    if not BED_PATH.exists():
        BED_PATH.mkdir(parents=True)
    get_all_chromosomes()
    
    chrom1 = decompress_chromosome("chr1")
    print(f"Length of chr1: {len(chrom1)}")
    print(f"Size of chr1: {sys.getsizeof(chrom1)} bytes")
    
    for bed_file in BED_FILES:
        bed_data = read_bed_file(bed_file)
        print(f"Read {bed_file} with shape: {bed_data.shape}")
        print(bed_data.head())
        print(bed_data.chrom.unique())
    
if __name__ == "__main__":
    main()