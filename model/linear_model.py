import pickle
from FlowDataTradeSystem.model.model_base import ModelBase
import numpy as np
import statsmodels.api as sm

class LinearModel(ModelBase):
    """线性模型类"""

    def __init__(self, model_path, input_features):
        self.model_path = model_path
        self.input_features = input_features
        self.model = self.load_model(model_path)

    @staticmethod
    def model_name():
        return "lin_model"

    def load_model(self, model_path):
        """加载线性回归模型"""
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def predict(self, factor_values):
        """使用线性模型进行预测"""
        input_data = np.array([factor_values[feature] for feature in self.input_features]).reshape(1, -1)

        params = self.model['params']
        X = sm.add_constant(input_data, has_constant=True)
        y_pred = np.dot(X, params)
        return y_pred.item()
