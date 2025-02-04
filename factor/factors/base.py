from abc import ABC, abstractmethod

class Factor(ABC):
    """因子基类"""
    _registry = []  # 存储所有因子子类

    def __init_subclass__(cls, **kwargs):
        """自动注册子类"""
        super().__init_subclass__(**kwargs)
        Factor._registry.append(cls)  # 注册子类

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.state = {}  # 用于存储因子状态


    @abstractmethod
    def compute(self, builder, **kwargs):
        pass


