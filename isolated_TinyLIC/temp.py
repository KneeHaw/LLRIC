def torch_test():
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version for PyTorch: {torch.version.cuda}")


def main():
    mac_count = 0
    with open("./saved/MACS.csv", "r") as f:
        lines = f.readlines()
        for line in lines:
            vals = line.strip().split(",")
            if len(vals) == 2 and vals[0] == "Linear":
                mac_count += int(vals[1])
    print(f"Total MACs for Linear layers: {mac_count:,}")
    
if __name__ == "__main__":
    main()