def torch_test():
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version for PyTorch: {torch.version.cuda}")


def main():
    timers = {"total": 32,
              "1": 5,
              "foo": 12,
              "ew": 0.43}
    total_t = timers["total"]
    # timers.pop("total")
    
    frac_times = {}
    for key, value in timers.items():
        
        frac_times[key] = value / total_t
        
    sorted_dict = dict(sorted(frac_times.items(), key=lambda item: item[1], reverse=True))
    for key, value in sorted_dict.items(): print(f"{key:<10} -> {value:.4f}")
    
    
    # foo = {"total": 0.0,
    #        "encode": 0.0,
    #        "entropy": 0.0,
    #        "decode": 0.0,
    #        "llric": 0.0}
        
if __name__ == "__main__":
    main()