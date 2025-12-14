from cas_reg import CAS
from pathlib import Path

cas_list = []

script_dir = Path(__file__).resolve().parent
print(script_dir)

with open(script_dir / "compounds.txt") as f:
    for line in f:
        cas_list.append(CAS(num=line.strip()))

cas_list.sort()

with open(script_dir / "cas_numbers_sorted.txt", "w") as f:
    for cas in cas_list:
        f.write(str(cas) + "\n")
        