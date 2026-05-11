import os
import subprocess
import sys
import gzip
import pandas as pd

from pathlib import Path

PATH = Path(__file__).resolve().parent.parent
DOWNLOAD_PATH = PATH / "sample" / "human_genome"
BED_PATH = PATH / "sample" / "bed_files"
CHROMOSOMES = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]

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

    with gzip.open(gz_path, "rt") as f:
        return f.read()


def read_bed_file(bed_file: str):
    bed_path = BED_PATH / bed_file
    if not bed_path.exists():
        raise FileNotFoundError(f"{bed_path} does not exist.")

    with gzip.open(bed_path, "rt") as f:
        data = pd.read_csv(f, sep="\t", header=None)
        data.columns = ["chrom", "start", "end", "state"]
    return data


def extract_binned_sequences(df: pd.DataFrame, bin_size: int = 200) -> pd.DataFrame:
    """
    Decompresses the run-length encoding of a BED dataframe into smaller bins
    and extracts the corresponding sequence from the human genome.
    """
    records = []
    
    # Group by chromosome to load one sequence at a time and save memory
    for chrom, group in df.groupby("chrom"):
        try:
            fasta_str = decompress_chromosome(chrom)
            # Remove the FASTA header and joined newlines to match 0-based indexing
            seq = "".join(fasta_str.split("\n")[1:])
        except FileNotFoundError:
            print(f"Warning: sequence for {chrom} not found, skipping...")
            continue
            
        for _, row in group.iterrows():
            start = row["start"]
            end = row["end"]
            state = row["state"]
            
            for chunk_start in range(start, end, bin_size):
                chunk_end = chunk_start + bin_size
                if chunk_end <= end:
                    chunk_seq = seq[chunk_start:chunk_end].upper()
                    if len(chunk_seq) == bin_size:
                        records.append({
                            "chrom": chrom,
                            "start": chunk_start,
                            "end": chunk_end,
                            "state": state,
                            "sequence": chunk_seq
                        })
                        
    return pd.DataFrame(records)


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
        print("Testing extract_binned_sequences on the first 10 rows:")
        
        # Test on a small subset to prevent long execution times
        subset = bed_data[bed_data['chrom'] == 'chr1'].head(10)
        print(subset)
        if not subset.empty:
            binned_df = extract_binned_sequences(subset, bin_size=200)
            print(f"Binned DataFrame shape: {binned_df.shape}")
            print(binned_df.head(200))
        else:
            print("No chr1 data found in the first rows to test.")
        
        break  # Just test on the first BED file


if __name__ == "__main__":
    main()
