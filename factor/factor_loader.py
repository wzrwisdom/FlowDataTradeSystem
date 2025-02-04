import os
import importlib

def load_factors_from_directory(directory: str, package: str):
    """
    动态加载指定目录下的所有因子模块
    参数:
        - directory: 因子子类所在的文件夹路径
        - package: 对应的 Python 包名
    """
    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("__"):
            # 获取模块名（去掉扩展名）
            module_name = filename[:-3]
            package_name = package.replace("/", ".")
            # 动态导入模块
            importlib.import_module(f"{package_name}.{module_name}")