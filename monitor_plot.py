# Plot the Loss and Top-1 while training.
import json
import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse


# 刷新频率（秒）
REFRESH_RATE = 10 

def parse_logs(log_path):
    metrics = defaultdict(list)
    
    if not os.path.exists(log_path):
        return metrics

    with open(log_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                step = data.get("global_step", data.get("epoch", 0))
                
                # 遍历提取我们需要画的指标
                for key, value in data.items():
                    if key == "loss" or "medusa" in key:
                        metrics[key].append((step, value))
            except:
                continue
    return metrics

def plot_metrics(metrics, save_path, args):
    if not metrics:
        return

    # 创建两个子图: 上面画 Loss，下面画 Top-1 Acc
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('attn_medusa Training Real-time Monitor', fontsize=16)

    # 1. 绘制 Loss 曲线
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Steps / Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制基础 loss
    if "loss" in metrics:
        steps, vals = zip(*metrics["loss"])
        ax1.plot(steps, vals, label="ToTal Loss", color="black", linewidth=2)
        
    width = args.smooth

    # 绘制各个 Medusa 头的 loss
    for key, pairs in metrics.items():
        if "loss" in key and key != "loss":
            steps, vals = zip(*pairs)

            if args.log_loss:
                ax1.set_yscale('log')

            # 因为 compute_loss 调用频繁，通过滑动窗口做个平滑处理使曲线更漂亮
            if len(vals) > width:
                vals_smooth = [sum(vals[max(0, i-width):i+1])/len(vals[max(0, i-width):i+1]) for i in range(len(vals))]
                ax1.plot(steps, vals_smooth, label=f"{key} (smooth width = {width})", alpha=0.8)
            else:
                ax1.plot(steps, vals, label=key, alpha=0.8)
    
                
    ax1.legend()

    # 2. 绘制 Top-1 Accuracy 曲线
    ax2.set_title("Medusa Heads Top-1 Accuracy")
    ax2.set_xlabel("Steps / Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    has_acc = False
    for key, pairs in metrics.items():
        if "top1" in key:
            has_acc = True
            steps, vals = zip(*pairs)

            # 滑动平均平滑
            if len(vals) > width:
                vals_smooth = [sum(vals[max(0, i-width):i+1])/len(vals[max(0, i-width):i+1]) for i in range(len(vals))]
                ax2.plot(steps, vals_smooth, label=f"{key} (smooth width = {width})", alpha=0.8)
            else:
                ax2.plot(steps, vals, label=key, alpha=0.8)
                
    if has_acc:
        ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot")
    parser.add_argument("--smooth", type=int, default=1, help="The smooth width.")
    parser.add_argument("--log_loss", action="store_true", help="use log scale of loss.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    TASKS = [
        (
            "/home/pzli/Project/Spec/SpS/260327_attn_medusa/commit/output/output_qwen3_8b_20260329_011616_lr_5e-4/training_logs.jsonl",
            "/home/pzli/Project/Spec/SpS/260327_attn_medusa/commit/output/output_qwen3_8b_20260329_011616_lr_5e-4/training_curves.png"
        ),
    ]

    print(f"🚀 Started monitoring {len(TASKS)} tasks...")
    print(f"📊 Refresh every {REFRESH_RATE} seconds")

    while True:
        try:
            for log_path, save_path in TASKS:
                try:
                    metrics = parse_logs(log_path)
                    if metrics:
                        plot_metrics(metrics, save_path, args)
                        print(f"✅ Updated: {save_path}")
                except Exception as e:
                    print(f"❌ Error in {log_path}: {e}")

            time.sleep(REFRESH_RATE)

        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
