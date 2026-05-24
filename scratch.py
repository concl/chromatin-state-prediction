
from templates.data import get_bed_files
from pathlib import Path

bed_file = "IHECRE00000994.7_18_ChromHMM.bed.gz"

get_bed_files(file_names=[bed_file])