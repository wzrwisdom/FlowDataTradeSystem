from abc import ABC, abstractmethod

class ModelBase(ABC):
    _registry = {}  # 注册所有模型类

    def __init_subclass__(cls, **kwargs):
        """自动注册子类"""
        super().__init_subclass__(**kwargs)
        ModelBase._registry[cls.model_name()] = cls

    @staticmethod
    @abstractmethod
    def model_name():
        """返回模型名称"""
        pass

    @abstractmethod
    def load_model(self, model_path):
        """加载模型"""
        pass

    @abstractmethod
    def predict(self, factor_values):
        """使用模型进行预测"""
        pass

    @classmethod
    def create_model(cls, model_type, model_path, input_features):
        """
        动态实例化模型类
        参数:
            - model_type: 模型类型字符串（如 'lin_model', 'CNN'）
            - model_path: 模型文件路径
            - input_features: 输入因子名称列表
        返回:
            - 模型实例
        """
        if model_type not in cls._registry:
            raise ValueError(f"未注册的模型类型: {model_type}")
        return cls._registry[model_type](model_path, input_features)
