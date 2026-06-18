from pathlib import Path
from collections import Counter

def count_classes(labels_dir):
    counter = Counter()
    for label_file in Path(labels_dir).glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                cls = line.strip().split()[0]
                counter[cls] += 1
    return counter

print(count_classes("Datasets/dataset2/train/labels"))  # check your FULL dataset, not the cut version