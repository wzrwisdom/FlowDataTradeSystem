from FlowDataTradeSystem.factor.factors.base import Factor

from loguru import logger as log
from collections import deque
import yaml
import time


class FactorBuilder:
    def __init__(self, feature_builder, config_filepath):
        self.feature_builder = feature_builder
        self.factors = []
        self.register_factors_by_config(config_filepath)
        self.history_facComb_bySym = {}

    def add_facComb(self, symbol, facComb):
        if symbol not in self.history_facComb_bySym.keys():
            self.history_facComb_bySym[symbol] = deque()
        else:
            self.history_facComb_bySym[symbol].append(facComb)

    def register_factors_by_config(self, config_file):
        """加载配置文件并注册因子"""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        factor_names = config.keys()
        for factor_class in Factor._registry:
            exist_flag = factor_class.name in factor_names
            if exist_flag:
                kwargs = config[factor_class.name]
                if kwargs is not None:
                    self.factors.append(factor_class(**kwargs))
                else:
                    log.debug(f"因子{factor_class.name}没有参数配置")  # 实际运行需要注释
                    self.factors.append(factor_class())


    def compute_all_factors(self):
        """
        计算所有注册的因子
        :return:
        """
        results = {}
        # computation_times = {}

        for factor in self.factors:
            factor_name = factor.name

            # start_time = time.perf_counter()
            result = factor.compute(self.feature_builder)
            # end_time = time.perf_counter()

            # 保存计算结果和耗时
            results[factor_name] = result
            # computation_times[factor_name] = end_time - start_time

        # # 打印或存储耗时信息
        # for factor_name, elapsed_time in computation_times.items():
        #     print(f"Factor '{factor_name}' computed in {elapsed_time:.4f} seconds")

        return results

    def preprocess(self, filepath, judge_col, no_winsorize_factors, factors):
        # 因子预处理的参数文件读取 可能不要放在这儿 避免重复加载
        with open(filepath, 'r') as f:
            params_dict = yaml.safe_load(f)

        fillZero_flag = factors[judge_col] is not None
        if not fillZero_flag:
            log.debug(f"因子{judge_col}为空")
            return None
        for factor_name in factors.keys():
            value = factors[factor_name]
            if (value is not None) and (factor_name in params_dict.keys()):
                params = params_dict[factor_name]
                if 'lower_quantile' in params.keys() and factor_name not in no_winsorize_factors:
                    lower_q = params['lower_quantile']
                    upper_q = params['upper_quantile']
                    if value < lower_q:
                        factors[factor_name] = lower_q
                    elif value > upper_q:
                        factors[factor_name] = upper_q
                if 'fac_mean' in params.keys() and 'fac_std' in params.keys():
                    fac_mean = params['fac_mean']
                    fac_std = params['fac_std']
                    factors[factor_name] = (factors[factor_name] - fac_mean) / fac_std

            if value is None and fillZero_flag:
                factors[factor_name] = 0

        return factors
