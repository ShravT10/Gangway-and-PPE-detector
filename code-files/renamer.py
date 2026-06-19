from pathlib import Path

labels_dir = Path("Datasets/person-only-dataset/valid/labels")

for txt_file in labels_dir.rglob("*.txt"):
    new_lines = []

    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) == 0:
                continue

            parts[0] = "2"  # Worker -> class 2

            new_lines.append(" ".join(parts))

    with open(txt_file, "w") as f:
        f.write("\n".join(new_lines))