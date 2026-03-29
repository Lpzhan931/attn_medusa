
# 环境
cuda 12.8.1
transformers==5.3.0
torch==2.8.0

# 执行训练
1. 从 HuggingFace 下载数据集 (Lpzhan/attn_medusa_train_data) 并放到仓库根目录下.
2. 设置 train.sh 中的 BASE_MODEL, DATASET 以及 torchrun 中的 GPU 数量.
3. 执行 bash train.sh

# 注意
1. 相比于 attn_medusa_model.py, attn_medusa_model_profile.py 结构类似, 但多了一些计时代码, 修改时两者需要**一致修改**.
2. 评测时需要修改代码中的路径.
3. 绘制训练日志曲线时需要修改路径.

