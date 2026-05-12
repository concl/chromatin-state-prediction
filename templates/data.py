import os
import subprocess
import sys
import gzip
import pandas as pd
from shutil import copyfileobj

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

LABELS = [
    "1_TssA",
    "2_TssFlnk",
    "3_TssFlnkU",
    "4_TssFlnkD",
    "5_Tx",
    "6_TxWk",
    "7_EnhG1",
    "8_EnhG2",
    "9_EnhA1",
    "10_EnhA2",
    "11_EnhWk",
    "12_ZNF/Rpts",
    "13_Het",
    "14_TssBiv",
    "15_EnhBiv",
    "16_ReprPC",
    "17_ReprPCWk",
    "18_Quies",
]
LABEL_TO_INDEX = {
    "1_TssA": 0,
    "2_TssFlnk": 1,
    "3_TssFlnkU": 2,
    "4_TssFlnkD": 3,
    "5_Tx": 4,
    "6_TxWk": 5,
    "7_EnhG1": 6,
    "8_EnhG2": 7,
    "9_EnhA1": 8,
    "10_EnhA2": 9,
    "11_EnhWk": 10,
    "12_ZNF/Rpts": 11,
    "13_Het": 12,
    "14_TssBiv": 13,
    "15_EnhBiv": 14,
    "16_ReprPC": 15,
    "17_ReprPCWk": 16,
    "18_Quies": 17,
}

SOURCE = "ftp://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/"


def get_all_chromosomes(download_path: Path = DOWNLOAD_PATH):
    """
    Downloads the FASTA files for all human chromosomes from the UCSC Genome Browser
    and saves them to the DOWNLOAD_PATH directory.
    """
    for chrom in CHROMOSOMES:
        url = f"{SOURCE}{chrom}.fa.gz"
        output_path = DOWNLOAD_PATH / f"{chrom}.fa.gz"
        if not output_path.exists():
            print(f"Downloading {chrom}...")
            subprocess.run(["curl", "-o", str(output_path), url], check=True)
        else:
            print(f"{chrom} already exists, skipping download.")


def decompress_chromosome(chrom: str) -> str:
    """
    Decompresses the FASTA file for a given chromosome and returns the sequence as a string.
    """
    gz_path = DOWNLOAD_PATH / f"{chrom}.fa.gz"
    if not gz_path.exists():
        raise FileNotFoundError(f"{gz_path} does not exist.")

    with gzip.open(gz_path, "rt") as f:
        return f.read()


def read_bed_file(bed_file: str) -> pd.DataFrame:
    """
    Reads a gzipped BED file and returns a DataFrame with columns: chrom, start, end, state.
    """

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
                        records.append(
                            {
                                "chrom": chrom,
                                "start": chunk_start,
                                "end": chunk_end,
                                "state": state,
                                "sequence": chunk_seq,
                            }
                        )

    return pd.DataFrame(records)


def gzip_file(input_path: Path, output_path: Path):
    with open(input_path, "rb") as f_in:
        with gzip.open(output_path, "wb") as f_out:
            copyfileobj(f_in, f_out)


def main():
    if not DOWNLOAD_PATH.exists():
        DOWNLOAD_PATH.mkdir(parents=True)
    if not BED_PATH.exists():
        BED_PATH.mkdir(parents=True)
    get_all_chromosomes()

    if not (PATH / "sample" / "binned_dataframe" / "test_binned.parquet").exists():
        print("Creating a sample binned DataFrame for the first BED file...")
        (PATH / "sample" / "binned_dataframe").mkdir(parents=True, exist_ok=True)
        test_bed_file = BED_FILES[0]
        bed_data = read_bed_file(test_bed_file)
        binned_df = extract_binned_sequences(bed_data, bin_size=200)
        binned_df.to_parquet(
            PATH / "sample" / "binned_dataframe" / "test_binned.parquet", index=False
        )

    if not (PATH / "sample" / "binned_dataframe" / "test_binned.parquet.gz").exists():
        print("Compressing the sample binned DataFrame...")
        (PATH / "sample" / "binned_dataframe").mkdir(parents=True, exist_ok=True)
        gzip_file(
            PATH / "sample" / "binned_dataframe" / "test_binned.parquet",
            PATH / "sample" / "binned_dataframe" / "test_binned.parquet.gz",
        )
    
    df = pd.read_parquet(PATH / "sample" / "binned_dataframe" / "test_binned.parquet")
    print(df.head())

if __name__ == "__main__":
    main()
