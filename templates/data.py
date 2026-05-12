import os
import socket
import subprocess
import sys
import gzip
import pandas as pd
from shutil import copyfileobj
import numpy as np
from paramiko import AutoAddPolicy, SSHClient
from dotenv import load_dotenv

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

def get_bed_files(n_files: int = 10) -> list[str]:
    """
    Downloads up to `n_files` BED files from a remote server via SFTP.

    Expects the environment variable `BED_FILES_REMOTE_PATH` to be set in the
    format `hostname:/path/to/remote/dir/`.

    Returns a list of filenames that were downloaded (or already present locally).
    """
    load_dotenv()  # Load environment variables from .env file
    remote_path = os.getenv("BED_FILES_REMOTE_PATH")
    if not remote_path:
        raise ValueError(
            "Environment variable BED_FILES_REMOTE_PATH is not set. "
            "Set it to e.g. 'myhost:/data/bed_files/'"
        )

    # Ensure the remote directory path ends with a separator for safe joining
    if ":" not in remote_path:
        raise ValueError(
            f"BED_FILES_REMOTE_PATH must be in 'host:/path/' format, got: {remote_path!r}"
        )

    hostname, remote_dir = remote_path.split(":", 1)
    if not remote_dir.endswith("/"):
        remote_dir += "/"

    # Ensure local BED path exists
    BED_PATH.mkdir(parents=True, exist_ok=True)

    ssh_username = os.getenv("SSH_USERNAME")
    ssh_password = os.getenv("SSH_PASSWORD")

    client = SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(AutoAddPolicy())

    sftp = None
    try:
        client.connect(
            hostname,
            username=ssh_username,
            password=ssh_password,
            timeout=10,
        )
        sftp = client.open_sftp()

        # List and select up to n_files from the remote directory
        remote_files = sftp.listdir(remote_dir)
        selected_files = remote_files[:n_files]

        downloaded: list[str] = []
        for file in selected_files:
            local_path = BED_PATH / file
            if local_path.exists():
                print(f"{file} already exists locally, skipping download.")
            else:
                print(f"Downloading {file} from remote server...")
                sftp.get(remote_dir + file, str(local_path))
            downloaded.append(file)

        return downloaded

    except socket.gaierror:
        raise ConnectionError(
            f"Cannot resolve hostname '{hostname}'. "
            f"This server may only be reachable from a specific network (e.g., VPN or institutional network)."
        )
    except OSError as e:
        raise ConnectionError(
            f"Cannot connect to '{hostname}': {e}. "
            f"Check that the host is reachable and you are on the correct network/VPN."
        )
    finally:
        if sftp is not None:
            sftp.close()
        client.close()
    
    
    

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



def extract_long_sequences(df: pd.DataFrame, window_size: int = 196608, stride: int = 98304, bin_size: int = 128) -> pd.DataFrame:
    """
    Extracts rolling contiguous sequences of size window_size, shifted by stride.
    For each sequence, it creates an array of length window_size // bin_size, where
    each element is the most common state annotation in that bin_size chunk.
    Only keeps sequences where at least 95% of the region is annotated.
    """
    records = []
    
    # We group by chromosome to load the genomic sequence into memory exactly once per chromosome.
    for chrom, group in df.groupby("chrom"):
        try:
            fasta_str = decompress_chromosome(chrom)
            # Remove the FASTA header and joined newlines to match 0-based indexing
            seq = "".join(fasta_str.split("\n")[1:]).upper()
        except FileNotFoundError:
            print(f"Warning: sequence for {chrom} not found, skipping...")
            continue

        seq_len = len(seq)
        
        # Create a dense array for the entire chromosome track
        # Initialize with 0 (unannotated background)
        state_array = np.zeros(seq_len, dtype=np.int16)
        
        states = group["state"]
        if not pd.api.types.is_numeric_dtype(states):
            states = states.astype(str).str.extract(r'(\d+)', expand=False).fillna(0).astype(int)
        
        # Fast array assignment map representing sequence locations
        for start, end, state in zip(group["start"], group["end"], states):
            start = max(0, start)
            end = min(seq_len, end)
            if start < end:
                state_array[start:end] = state
                
        # Slide window over the chromosome
        for w_start in range(0, seq_len - window_size + 1, stride):
            w_end = w_start + window_size
            window_states = state_array[w_start:w_end]
            
            # Skip regions that mostly contain padding/unknown states
            if np.sum(window_states == 0) > (window_size * 0.05):
                continue
                
            chunk_seq = seq[w_start:w_end]
            
            # Subdivide 196,608 length into 128bp bins -> Total of 1,536 bins.
            # Shape is (1536, 128)
            reshaped_states = window_states.reshape(-1, bin_size)
            
            # Fast most-frequent label per bin computation
            # Using bincount to get the index of max occurrences in each 128bp bin
            binned_labels = [int(np.bincount(row).argmax()) for row in reshaped_states]
            
            records.append(
                {
                    "chrom": chrom,
                    "start": w_start,
                    "end": w_end,
                    "sequence": chunk_seq,
                    # We store it as a list so parquet can serialize it properly
                    "labels": binned_labels,
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
    
    df = pd.read_parquet(PATH / "sample" / "binned_dataframe" / "test_binned.parquet")
    print(df.head())

def generate_shards():
    train_dir = PATH / "sample" / "binned_dataframe" / "train_shards"
    val_dir = PATH / "sample" / "binned_dataframe" / "val_shards"
    
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
    else:
        print("No valid sequences found for validation chromosomes, skipping...")
        print(val_records.head())  # Debugging output for empty records
        
    print("Extracting Train Sequences (Other)...")
    # For memory safety, we process train chromosomes one by one
    for chrom, group in train_df.groupby("chrom"):
        print(f"  Extracting train sequence for {chrom}...")
        chrom_records = extract_long_sequences(group)
        if not chrom_records.empty:
            print(f"  Saving {chrom} to parquet...")
            chrom_records.to_parquet(train_dir / f"train_{chrom}.parquet")
        else:
            print(f"  No valid sequences found for {chrom}, skipping...")
            print(chrom_records.head())  # Debugging output for empty records

if __name__ == "__main__":
    main()
