from sortedcontainers import SortedDict
import datetime

class FeatureBuilder:
    def __init__(self):
        self.history_data = SortedDict()  # 使用 SortedDict 存储历史数据，自动保持排序

    def add_feature(self, timestamp, features):
        # 向 history_data 中添加特征值
        self.history_data[timestamp] = features

    def get_nearest_feature(self, target_timestamp):
        # 使用 bisect_right 找到最大的时间戳小于等于目标时间戳
        closest_timestamp = self.history_data.bisect_right(target_timestamp) - 1

        if closest_timestamp == -1:
            # 如果找不到比目标时间戳小的记录，则返回 None 或者某个默认值
            return None
        else:
            # 返回最接近的历史数据
            closest_timestamp = list(self.history_data.keys())[closest_timestamp]
            return self.history_data[closest_timestamp]


feature_builder = FeatureBuilder()

# 添加历史数据
feature_builder.add_feature(datetime.datetime(2023, 9, 25, 10, 0, 0), {"price": 100, "volume": 500})
feature_builder.add_feature(datetime.datetime(2023, 9, 25, 10, 5, 0), {"price": 102, "volume": 550})
feature_builder.add_feature(datetime.datetime(2023, 9, 25, 10, 10, 0), {"price": 105, "volume": 600})

# 获取与给定时间戳最接近的特征数据，且时间戳不能晚于目标时间
target_timestamp = datetime.datetime(2023, 9, 25, 9, 15, 0)
nearest_feature = feature_builder.get_nearest_feature(target_timestamp)

if nearest_feature:
    print("最接近的特征数据：", nearest_feature)
else:
    print("没有找到符合条件的历史数据")
