
# 环境
cuda 12.8.1
transformers==5.3.0
torch==2.8.0

# 执行训练
1. 从 HuggingFace 下载数据集 (Lpzhan/attn_medusa_train_data) 并放到仓库根目录下.
2. 设置 train.sh 中的 BASE_MODEL, DATASET 以及 torchrun 中的 GPU 数量.
3. 执行 bash train.sh
