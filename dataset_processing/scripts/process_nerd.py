def main():
    src_path = "data/Few-NERD/supervised"
    tgt_path = "data/Few-NERD"
    for split in ["train.txt", "dev.txt", "test.txt"]:
        with open(f"{src_path}/{split}", 'r') as reader:
            with open(f"{tgt_path}/{split}", 'w') as writer:
                for line in reader:
                    if line.strip() != "":
                        n_line = line.strip().split('\t')
                        if n_line[-1] != "O":
                            line = f"{n_line[0]}\tI-{n_line[-1]}\n"
                    writer.write(line)


if __name__ == "__main__":
    main()