import torch


def check_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"\n[详细硬件信息]")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")

        print("\n[性能测试]")
        tensor = torch.randn(10000, 10000).cuda()
        result = tensor * tensor
        print(f"GPU计算测试完成! 结果矩阵大小: {result.size()}")
    else:
        print("\n[!] 配置异常检查建议:")
        if not torch.version.cuda:
            print(" - 检测到CPU版本PyTorch，请安装GPU版本")
        else:
            print(" - CUDA驱动可能未正确安装")


if __name__ == "__main__":
    check_gpu()