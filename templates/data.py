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
ENFORMER_SEQUENCES_PATH = PATH / "sample" / "enformer_sequences"
CHROMOSOMES = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]

BED_FILES = [
    "IHECRE00001475.1_18_ChromHMM.bed.gz",
    "IHECRE00002659.1_18_ChromHMM.bed.gz",
    "IHECRE00002660.1_18_ChromHMM.bed.gz",
    "IHECRE00004909.1_18_ChromHMM.bed.gz",
    "IHECRE00004910.1_18_ChromHMM.bed.gz",
]

CHROMOSOME_SOURCE = "ftp://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/"


def get_bed_files(n_files: int = 0, file_names: list[str] = None) -> list[str]:
    """Download BED files from a remote server via SFTP.

    Expects the environment variable ``BED_FILES_REMOTE_PATH`` to be set in the
    format ``hostname:/path/to/remote/dir/``.

    Args:
        n_files: Maximum number of files to download. If 0, downloads all
            available files. Ignored if ``file_names`` is provided.
        file_names: Specific filenames to download. Overrides ``n_files``.

    Returns:
        List of filenames that were downloaded (or already present locally).

    Raises:
        ValueError: If ``BED_FILES_REMOTE_PATH`` is not set or has invalid format.
        ConnectionError: If the remote host cannot be resolved or reached.
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
        if not file_names:
            selected_files = remote_files[:n_files] if n_files > 0 else remote_files
        else:
            selected_files = [f for f in remote_files if f in file_names]
        
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
    """Download FASTA files for all human chromosomes from UCSC Genome Browser.

    Saves compressed ``.fa.gz`` files to ``download_path``. Skips chromosomes
    that already exist locally.

    Args:
        download_path: Directory where chromosome FASTA files are saved.
            Defaults to ``DOWNLOAD_PATH``.
    """
    for chrom in CHROMOSOMES:
        url = f"{CHROMOSOME_SOURCE}{chrom}.fa.gz"
        output_path = DOWNLOAD_PATH / f"{chrom}.fa.gz"
        if not output_path.exists():
            print(f"Downloading {chrom}...")
            subprocess.run(["curl", "-o", str(output_path), url], check=True)
        else:
            print(f"{chrom} already exists, skipping download.")


def decompress_chromosome(chrom: str) -> str:
    """Decompress a chromosome FASTA file and return its sequence as a string.

    Args:
        chrom: Chromosome name (e.g. ``"chr1"``).

    Returns:
        The full genomic sequence as a string.

    Raises:
        FileNotFoundError: If the compressed FASTA file does not exist locally.
    """
    gz_path = DOWNLOAD_PATH / f"{chrom}.fa.gz"
    if not gz_path.exists():
        raise FileNotFoundError(f"{gz_path} does not exist.")

    with gzip.open(gz_path, "rt") as f:
        return f.read()


def read_bed_file(bed_file: str) -> pd.DataFrame:
    """Read a gzipped BED file into a DataFrame.

    Args:
        bed_file: Filename of the gzipped BED file located in ``BED_PATH``.

    Returns:
        DataFrame with columns: ``chrom``, ``start``, ``end``, ``state``.

    Raises:
        FileNotFoundError: If the BED file does not exist locally.
    """

    bed_path = BED_PATH / bed_file
    if not bed_path.exists():
        bed_path = PATH / bed_file
        if not bed_path.exists():
            raise FileNotFoundError(f"{bed_file} not found in {BED_PATH} or {PATH}")

    with gzip.open(bed_path, "rt") as f:
        data = pd.read_csv(f, sep="\t", header=None)
        data.columns = ["chrom", "start", "end", "state"]
    return data


# hg38 chromosome sizes (from UCSC, used for bounds checking)
HG38_CHROM_SIZES: dict[str, int] = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468, "chrX": 156040895,
}


def extend_bed_intervals(
    input_bed: str | Path,
    output_bed: str | Path,
    extension_bp: int = 32768,
    chrom_sizes: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Extend BED intervals symmetrically and clip to chromosome boundaries.

    Reads a BED file (non-gzipped, TSV format), extends each interval by
    ``extension_bp`` on both the left and right, clips to ``[0, chr_len]``
    bounds, and writes the result to a new BED file.  For example, 131,072 bp
    intervals become 196,608 bp when ``extension_bp=32768`` (the Enformer
    default input length).

    Intervals that extend beyond chromosome boundaries are clipped.  If an
    interval cannot reach the target length after clipping (i.e. the
    chromosome is too short), a warning is printed and those rows are
    **excluded** from the output.

    Args:
        input_bed: Path to the input BED file.
        output_bed: Path where the extended BED file will be written.
        extension_bp: Number of base pairs to add to each side.
        chrom_sizes: Dictionary mapping chromosome names to their lengths.
            Defaults to ``HG38_CHROM_SIZES`` (hg38 assembly).

    Returns:
        DataFrame with columns ``chrom``, ``start``, ``end``, and any extra
        columns present in the input.
    """
    if chrom_sizes is None:
        chrom_sizes = HG38_CHROM_SIZES

    input_bed = Path(input_bed)
    output_bed = Path(output_bed)

    if not input_bed.exists():
        raise FileNotFoundError(f"{input_bed} does not exist.")

    df = pd.read_csv(input_bed, sep="\t", header=None)

    # BED files typically have at least 4 columns: chrom, start, end, ...
    df.columns = ["chrom", "start", "end"] + [
        f"col_{i}" for i in range(3, df.shape[1])
    ]

    target_len = df["end"].iloc[0] - df["start"].iloc[0] + 2 * extension_bp

    df["start"] = df["start"] - extension_bp
    df["end"] = df["end"] + extension_bp

    # Clip to chromosome boundaries
    df["chr_len"] = df["chrom"].map(chrom_sizes)

    missing = df["chr_len"].isna()
    if missing.any():
        missing_chroms = df.loc[missing, "chrom"].unique().tolist()
        raise ValueError(
            f"Chromosome sizes not found for: {missing_chroms}. "
            f"Add them to chrom_sizes or pass a custom dictionary."
        )

    before_clip = len(df)

    # Clip start to 0 and end to chromosome length
    df["start"] = df["start"].clip(lower=0)
    df["end"] = df["end"].clip(upper=df["chr_len"])

    # Warn about and drop intervals that can't reach the target length
    actual_len = df["end"] - df["start"]
    too_short = actual_len < target_len
    if too_short.any():
        print(
            f"Warning: {too_short.sum()} interval(s) cannot reach "
            f"{target_len:,} bp after clipping (chromosome boundary):"
        )
        for _, row in df[too_short].iterrows():
            print(
                f"  {row['chrom']}: [{row['start']:,}, {row['end']:,}) "
                f"= {actual_len[too_short].loc[row.name]:,} bp"
            )
        df = df[~too_short]

    df = df.drop(columns=["chr_len"])

    # Restore original column names (unnamed) for writing
    df.to_csv(output_bed, sep="\t", header=False, index=False)

    print(
        f"Extended {before_clip} intervals by ±{extension_bp:,} bp → "
        f"{len(df)} written to {output_bed} "
        f"(target {target_len:,} bp each)"
    )
    return df


def extract_binned_sequences(df: pd.DataFrame, bin_size: int = 200) -> pd.DataFrame:
    """Decompress BED run-length encoding into smaller bins with genomic sequences.

    Groups BED records by chromosome, loads each chromosome's FASTA once,
    then subdivides annotated regions into bins of ``bin_size`` bp and extracts
    the corresponding DNA sequence.

    Args:
        df: BED DataFrame with columns ``chrom``, ``start``, ``end``, ``state``.
        bin_size: Size of each bin in base pairs.

    Returns:
        DataFrame with columns: ``chrom``, ``start``, ``end``, ``state``,
        ``sequence``.
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


def extract_long_sequences(
    df: pd.DataFrame,
    window_size: int = 196608,
    stride: int = 128 * 896,
    bin_size: int = 128,
) -> pd.DataFrame:
    """Extract rolling contiguous sequences with per-bin chromatin state labels.

    Slides a window of ``window_size`` bp across each chromosome with step
    ``stride``. For each window, subdivides it into bins of ``bin_size`` bp and
    assigns each bin the majority chromatin state annotation. Only keeps windows
    where at least 95% of the region is annotated.

    Args:
        df: BED DataFrame with columns ``chrom``, ``start``, ``end``, ``state``.
        window_size: Total length of each extracted sequence in bp.
        stride: Step size between consecutive windows in bp.
        bin_size: Size of each label bin in bp (window_size must be divisible
            by bin_size).

    Returns:
        DataFrame with columns: ``chrom``, ``start``, ``end``, ``sequence``,
        ``labels``.
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
            states = (
                states.astype(str)
                .str.extract(r"(\d+)", expand=False)
                .fillna(0)
                .astype(int)
            )

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


def generate_shards(bed_file: str):
    train_dir = PATH / "sample" / "binned_dataframe" / "train_shards"
    val_dir = PATH / "sample" / "binned_dataframe" / "val_shards"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    val_chromosomes = ["chr8", "chr9"]

    # Process only the first BED file as an example to avoid OOM/Disk space issues
    # You would loop over all BED_FILES inside this loop in production
    print(f"Reading {bed_file}...")
    df = read_bed_file(bed_file)

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


def generate_shards_from_index(
    index_bed: str | Path = ENFORMER_SEQUENCES_PATH / "data_human_sequences_enformer.bed",
    annotation_bed_file: str | None = None,
    output_dir: str | Path | None = None,
    bin_size: int = 128,
) -> dict[str, Path]:
    """Generate train/valid/test shards using a predefined interval index.

    Uses ``data_human_sequences_enformer.bed`` (or another index BED) to look up
    exactly which 196,608 bp genomic intervals to extract.  For each interval,
    the DNA sequence is pulled from the chromosome FASTA and chromatin-state
    labels are derived from a ChromHMM annotation BED by intersecting the
    interval with the annotation track and majority-voting within each
    ``bin_size``-bp bin.

    Shards are saved as Parquet files under ``output_dir``, one per split::

        output_dir/
        ├── train_shards/
        │   ├── train_chr1.parquet
        │   ├── train_chr2.parquet
        │   └── ...
        ├── valid_shards/
        │   └── valid.parquet
        └── test_shards/
            └── test.parquet

    Args:
        index_bed: Path to the interval index BED file.  Must have columns
            ``chrom``, ``start``, ``end``, ``split`` where ``split`` is one of
            ``train``, ``valid``, ``test``.
        annotation_bed_file: Filename of the gzipped ChromHMM BED file in
            ``BED_PATH``.  Defaults to the first file in ``BED_FILES``.
        output_dir: Root directory for output shards.  Defaults to
            ``data/binned_dataframe/``.
        bin_size: Size of label bins in bp.  Must evenly divide the interval
            length (196,608).

    Returns:
        Dict mapping split name to its output directory path.
    """
    if annotation_bed_file is None:
        annotation_bed_file = BED_FILES[0]

    if output_dir is None:
        output_dir = PATH / "sample" / "binned_dataframe_enformer"
    output_dir = Path(output_dir)

    index_bed = Path(index_bed)
    if not index_bed.exists():
        raise FileNotFoundError(f"Index BED not found: {index_bed}")

    # ------------------------------------------------------------------
    # 1. Load the interval index
    # ------------------------------------------------------------------
    print(f"Loading interval index from {index_bed} ...")
    index_df = pd.read_csv(index_bed, sep="\t", header=None)
    index_df.columns = ["chrom", "start", "end", "split"]

    window_size = int(index_df["end"].iloc[0] - index_df["start"].iloc[0])
    if window_size % bin_size != 0:
        raise ValueError(
            f"window_size ({window_size}) must be divisible by bin_size ({bin_size})"
        )
    num_bins = window_size // bin_size

    splits = index_df["split"].unique()
    print(f"  {len(index_df)} intervals, window={window_size:,} bp, "
          f"bins={num_bins} × {bin_size} bp, splits={splits.tolist()}")

    # ------------------------------------------------------------------
    # 2. Load the ChromHMM annotation track
    # ------------------------------------------------------------------
    print(f"Loading annotation track: {annotation_bed_file} ...")
    annot_df = read_bed_file(annotation_bed_file)

    # Normalise state column to integer
    states = annot_df["state"]
    if not pd.api.types.is_numeric_dtype(states):
        states = (
            states.astype(str)
            .str.extract(r"(\d+)", expand=False)
            .fillna(0)
            .astype(int)
        )
    annot_df["state_int"] = states.astype(np.int16)

    # ------------------------------------------------------------------
    # 3. Process each chromosome once
    # ------------------------------------------------------------------
    for split_name in ["train", "valid", "test"]:
        split_dir = output_dir / f"{split_name}_shards"
        split_dir.mkdir(parents=True, exist_ok=True)

    all_chroms = index_df["chrom"].unique()
    for chrom in sorted(all_chroms):
        # Load chromosome sequence once
        try:
            fasta_str = decompress_chromosome(chrom)
            seq = "".join(fasta_str.split("\n")[1:]).upper()
        except FileNotFoundError:
            print(f"  Warning: FASTA for {chrom} not found, skipping.")
            continue

        seq_len = len(seq)
        chrom_index = index_df[index_df["chrom"] == chrom]
        chrom_annot = annot_df[annot_df["chrom"] == chrom]

        # Build a dense state array for this chromosome from annotations
        state_array = np.zeros(seq_len, dtype=np.int16)
        for _, row in chrom_annot.iterrows():
            s = max(0, int(row["start"]))
            e = min(seq_len, int(row["end"]))
            if s < e:
                state_array[s:e] = row["state_int"]

        # Collect records per split for this chromosome, then write once
        records_by_split: dict[str, list[dict]] = {
            "train": [], "valid": [], "test": []
        }
        kept = 0
        for _, interval in chrom_index.iterrows():
            w_start = int(interval["start"])
            w_end = int(interval["end"])
            split_name = interval["split"]

            if w_start < 0 or w_end > seq_len:
                continue

            window_states = state_array[w_start:w_end]

            # Skip if >5% of the window is unannotated
            if np.sum(window_states == 0) > (window_size * 0.05):
                continue

            chunk_seq = seq[w_start:w_end]
            if len(chunk_seq) != window_size:
                continue

            # Majority-vote binning
            reshaped = window_states.reshape(-1, bin_size)
            binned_labels = [
                int(np.bincount(row.astype(np.int32)).argmax())
                for row in reshaped
            ]

            records_by_split[split_name].append({
                "chrom": chrom,
                "start": w_start,
                "end": w_end,
                "sequence": chunk_seq,
                "labels": binned_labels,
            })
            kept += 1

        # Write one Parquet file per (split, chromosome)
        for split_name, recs in records_by_split.items():
            if not recs:
                continue
            split_dir = output_dir / f"{split_name}_shards"
            out_path = split_dir / f"{split_name}_{chrom}.parquet"
            pd.DataFrame(recs).to_parquet(out_path, index=False)

        skipped = len(chrom_index) - kept
        msg = f"  ✓ {chrom}: {kept} intervals written"
        if skipped:
            msg += f" ({skipped} skipped)"
        print(msg)

    print("Done.")
    return {
        "train": output_dir / "train_shards",
        "valid": output_dir / "valid_shards",
        "test": output_dir / "test_shards",
    }

# Human-readable ChromHMM state names (1-indexed)
STATE_NAMES = [
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


def merge_bed_intervals(
    predictions,
    state_names: list[str] | None = None,
) -> pd.DataFrame:
    """Merge adjacent same-state genomic bins into BED intervals.

    Takes an iterable of ``(chrom, start, end, state)`` tuples (sorted by
    position) and collapses consecutive bins that share the same chromosome
    and state into larger intervals, producing a compact BED-style annotation.

    Args:
        predictions: Iterable of ``(chrom, start, end, state)`` tuples.
        state_names: Optional list mapping 1-indexed state integers to
            human-readable names (e.g. ``"1_TssA"``).  If provided, the
            returned DataFrame will have a ``name`` column.

    Returns:
        DataFrame with columns ``chrom``, ``start``, ``end``, ``state``
        (and optionally ``name``).
    """
    records = []
    current = None  # (chrom, start, end, state)

    for chrom, start, end, state in predictions:
        if current is None:
            current = [chrom, start, end, state]
        elif current[0] == chrom and current[3] == state and current[2] == start:
            # Extend the current interval
            current[2] = end
        else:
            records.append(tuple(current))
            current = [chrom, start, end, state]

    if current is not None:
        records.append(tuple(current))

    df = pd.DataFrame(records, columns=["chrom", "start", "end", "state"])
    df["state"] = df["state"].astype(int)

    if state_names is not None:
        df["name"] = df["state"].apply(lambda s: state_names[s - 1])

    return df


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


if __name__ == "__main__":
    main()
