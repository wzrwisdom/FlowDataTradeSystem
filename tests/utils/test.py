import pandas as pd

# 创建一个示例 DataFrame
data = {
    'time': ['2023-09-25 09:30:00', '2023-09-25 09:31:00', '2023-09-25 09:32:00'],
    'price': [100, 101, 102],
    'volume': [1000, 1500, 1200],
}
snap_df = pd.DataFrame(data)

# 使用 itertuples
snap_tuples = snap_df.itertuples(index=False)  # 返回 namedtuple

# 将 namedtuple 转换回 DataFrame
snap_df_converted = pd.DataFrame([tuple(row) for row in snap_tuples], columns=snap_df.columns)

print("原始 DataFrame:")
print(snap_df)

print("\n通过 itertuples 转换回的 DataFrame:")
print(snap_df_converted)