"""
python benchmark_ar_qwen3.py \
    --base-model-path /home/share/models/Qwen3-8B/ \
    --bench-name gsm8k \
    --num-samples 1
"""

import torch
import time
import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

THINKING = False
DATA_DIR = "./data"       # from eagle repo

@torch.no_grad()
def benchmark_ar_generate(model, tokenizer, input_ids, max_new_tokens=512):
    device = model.device

    stats = {
        "decode_time": 0,
        "new_tokens": 0
    }

    terminators = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and isinstance(im_end, int):
            terminators.append(im_end)

    seq = input_ids[0].tolist()

    # ==========================================
    # 1. Prefill（不计入时间，与 SpS 对齐）
    # ==========================================
    out = model(input_ids.to(device), use_cache=True)
    kv = out.past_key_values

    token = torch.argmax(out.logits[0, -1, :]).item()
    seq.append(token)
    stats["new_tokens"] += 1

    # ==========================================
    # 2. AR Decode（开始计时）
    # ==========================================
    while stats["new_tokens"] < max_new_tokens:
        if token in terminators:
            break

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        out = model(
            torch.tensor([[token]], device=device),
            past_key_values=kv,
            use_cache=True
        )
        kv = out.past_key_values

        token = torch.argmax(out.logits[0, -1, :]).item()
        seq.append(token)

        torch.cuda.synchronize()
        stats["decode_time"] += (time.perf_counter() - t0)

        stats["new_tokens"] += 1

    return stats


def run_benchmark(args):
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    # 加载数据（与 SpS 完全一致）
    questions = []
    question_file = os.path.join(DATA_DIR, f"{args.bench_name}/question.jsonl")
    with open(question_file, "r") as f:
        for line in f:
            questions.append(json.loads(line))

    # ==========================================
    # 1. Warmup（对齐 SpS）
    # ==========================================
    print("🔥 Warming up...")
    dummy_input = tokenizer(
        ["Hello, who are you?"],
        return_tensors="pt"
    ).input_ids.to(model.device)

    benchmark_ar_generate(model, tokenizer, dummy_input, max_new_tokens=16)

    # ==========================================
    # 2. 正式评测
    # ==========================================
    print("🚀 Starting AR benchmark...")

    all_stats = []

    for q in tqdm(questions[:args.num_samples], total=args.num_samples):
        messages = [{"role": "user", "content": q["turns"][0]}]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=THINKING
        )

        input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(model.device)

        start_w = time.perf_counter()

        stat = benchmark_ar_generate(
            model,
            tokenizer,
            input_ids,
            max_new_tokens=args.max_new_tokens
        )

        stat["total_wall_time"] = time.perf_counter() - start_w
        all_stats.append(stat)

    # ==========================================
    # 3. 统计（对齐 SpS 输出格式）
    # ==========================================
    total_tokens = sum(s["new_tokens"] for s in all_stats)
    total_time = sum(s["total_wall_time"] for s in all_stats)

    speed = total_tokens / total_time

    decode_time = sum(s["decode_time"] for s in all_stats)

    print(f"\n====== AR 测试结果 ======")
    print(f"Model: {args.base_model_path}")
    print(f"1. 解码速度: {speed:.2f} tokens/s")
    print(f"2. Decode 时间占比: {decode_time / total_time * 100:.1f}%")
    print(f"3. 平均每 token 时延: {decode_time / total_tokens * 1000:.2f} ms")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--bench-name", type=str, default="gsm8k")
    parser.add_argument("--num-samples", type=int, default=10)

    args = parser.parse_args()
    run_benchmark(args)
