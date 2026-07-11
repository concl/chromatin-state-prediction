
from templates.data import (
    PATH,
    DOWNLOAD_PATH,
    BED_PATH,
    BED_FILES,
    get_all_chromosomes,
    read_bed_file,
    extract_binned_sequences,
)
import pandas as pd

def main():
    if not DOWNLOAD_PATH.exists():
        DOWNLOAD_PATH.mkdir(parents=True)
    if not BED_PATH.exists():
        BED_PATH.mkdir(parents=True)
    get_all_chromosomes()

    if not (PATH / "data" / "binned_dataframe" / "test_binned.parquet").exists():
        print("Creating a sample binned DataFrame for the first BED file...")
        (PATH / "data" / "binned_dataframe").mkdir(parents=True, exist_ok=True)
        test_bed_file = BED_FILES[0]
        bed_data = read_bed_file(test_bed_file)
        binned_df = extract_binned_sequences(bed_data, bin_size=200)
        binned_df.to_parquet(
            PATH / "data" / "binned_dataframe" / "test_binned.parquet", index=False
        )

    df = pd.read_parquet(PATH / "data" / "binned_dataframe" / "test_binned.parquet")
    print(df.head())


if __name__ == "__main__":
    main()
