import time


def timer(func):
    """计时器装饰器，记录函数执行时间（秒）"""

    def wrapper(*args, **kwargs):
        # 记录开始时间
        start_time = time.perf_counter()

        # 执行原函数
        result = func(*args, **kwargs)

        # 计算并打印耗时
        duration = time.perf_counter() - start_time
        print(f"函数 {func.__name__} 执行耗时: {duration:.6f} 秒")

        return result

    return wrapper