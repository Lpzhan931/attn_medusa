"""
python benchmark_attn_medusa.py \
    --medusa-model-path ./output/output_qwen3_8b_20260329_011616_lr_5e-4/checkpoint-384 \
    --base-model-path /home/share/models/Qwen3-8B/ \
    --bench-name gsm8k \
    --num-samples 1 \
    --gamma 4 \
    --show-first-sample

python benchmark_attn_medusa.py \
    --medusa-model-path ./output/output_qwen3_8b_20260329_011616_lr_5e-4/checkpoint-384 \
    --base-model-path /home/share/models/Qwen3-8B/ \
    --bench-name gsm8k \
    --num-samples 10 \
    --gamma 4
    
"""

import torch
import time
import json
import os
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer

# from attn_medusa_model import AttnMedusaModel
from attn_medusa_model_profile import AttnMedusaModel

THINKING = False
DATA_DIR = "./data"
LOG_DIR = "./evaluation/logs"

def trim_kv_cache(past_key_values, keep_len):
    """
    安全裁剪 KV Cache，兼容新版 DynamicCache 与 Tuple
    """
    if past_key_values is None: 
        return None
    
    # 1. Transformers 新版 DynamicCache (最佳做法：调用官方 crop)
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(keep_len)
        return past_key_values
    
    # 2. 传统 tuple 格式 (基础模型的多层)
    if isinstance(past_key_values, tuple) and isinstance(past_key_values[0], tuple):
        new_past = []
        for layer in past_key_values:
            k = layer[0][:, :, :keep_len, :]
            v = layer[1][:, :, :keep_len, :]
            new_past.append((k, v))
        return tuple(new_past)
        
    # 3. 单层 tuple 格式 (AttnMedusa 的 decoder_layer 缓存)
    if isinstance(past_key_values, tuple) and torch.is_tensor(past_key_values[0]):
        k = past_key_values[0][:, :, :keep_len, :]
        v = past_key_values[1][:, :, :keep_len, :]
        return (k, v)
        
    return past_key_values

@torch.no_grad()
def benchmark_attn_medusa_generate(model, tokenizer, input_ids, max_new_tokens=512, gamma=4, debug_log_file=None):
    device = model.base_model.device
    
    stats = {
        "forward_time": 0,  
        "forward_count": 0,  
        "accepted_lengths": [],
        "position_attempts": torch.zeros(gamma).to(device), 
        "position_accepts": torch.zeros(gamma).to(device),
        "new_tokens": 0
    }

    terminators = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and isinstance(im_end, int):
            terminators.append(im_end)

    prompt_len = input_ids.shape[1]
    seq = input_ids[0].tolist()
    
    if debug_log_file:
        debug_log_file.write("="*60 + "\n")
        debug_log_file.write(f"🔍 [DEBUG] Attn-Medusa 投机解码详细过程追踪\n")
        debug_log_file.write("="*60 + "\n")
        debug_log_file.write(f"📝 Prompt 长度: {prompt_len} tokens\n\n")

    # ==========================================
    # 0. 预填充 (Prefill)
    # ==========================================
    t0 = time.perf_counter()
    keep_len = prompt_len
    position_ids = torch.arange(0, keep_len, dtype=torch.long, device=device).unsqueeze(0)
    
    # Prefill 时必须传递 full attention_mask 避免 decoder_layer 变成双向注意力
    attention_mask = torch.ones_like(input_ids)
    
    # m_logits_stack, outputs, orig_logits, medusa_kv = model(
    #     input_ids=input_ids.to(device),
    #     attention_mask=attention_mask,
    #     position_ids=position_ids,
    #     output_orig=True
    # )

    m_logits_stack, outputs, orig_logits, medusa_kv, latencies = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_orig=True,
        return_latencies=True   # 开启精细测时
    )
    
    stats["time_base"] = stats.get("time_base", 0) + latencies["base_model"]
    stats["time_decoder"] = stats.get("time_decoder", 0) + latencies["decoder_layer"]
    stats["time_medusa"] = stats.get("time_medusa", 0) + latencies["medusa_heads"]
    stats["counts"] = stats.get("counts", 0) + 1


    base_kv = outputs.past_key_values
    
    # 第一步预测的 Token
    next_token = torch.argmax(orig_logits[0, -1, :]).item()
    seq.append(next_token)
    stats["new_tokens"] += 1
    
    # 初始化草稿
    draft_tokens = torch.argmax(m_logits_stack[0, :gamma, -1, :], dim=-1).tolist()
    
    stats["forward_time"] += (time.perf_counter() - t0)
    stats["forward_count"] += 1
    step_counter = 0

    # ==========================================
    # 1. 投机解码主循环
    # ==========================================
    while stats["new_tokens"] < max_new_tokens:
        if seq[-1] in terminators:
            break
        step_counter += 1
        
        torch.cuda.synchronize()
        t_step_start = time.perf_counter()
        
        curr_input_list = [next_token] + draft_tokens
        curr_in = torch.tensor([curr_input_list], device=device)
        
        curr_len = len(curr_input_list)
        position_ids = torch.arange(keep_len, keep_len + curr_len, dtype=torch.long, device=device).unsqueeze(0)
        
        # 在 Generation 时 attention_mask 传 None，交由底层自动处理因果掩码和 KV Cache
        # m_logits_stack, outputs, orig_logits, medusa_kv = model(
        #     input_ids=curr_in,
        #     attention_mask=None, 
        #     past_key_values=base_kv,
        #     medusa_past_key_values=medusa_kv,
        #     position_ids=position_ids,
        #     output_orig=True
        # )

        m_logits_stack, outputs, orig_logits, medusa_kv, latencies = model(
            input_ids=curr_in,
            attention_mask=None, 
            past_key_values=base_kv,
            medusa_past_key_values=medusa_kv,
            position_ids=position_ids,
            output_orig=True,
            return_latencies=True   # 开启精细测时
        )
        
        stats["time_base"] = stats.get("time_base", 0) + latencies["base_model"]
        stats["time_decoder"] = stats.get("time_decoder", 0) + latencies["decoder_layer"]
        stats["time_medusa"] = stats.get("time_medusa", 0) + latencies["medusa_heads"]
        stats["counts"] = stats.get("counts", 0) + 1


        base_kv = outputs.past_key_values
        
        target_preds = torch.argmax(orig_logits[0], dim=-1).tolist()
        
        # 验证阶段 (对齐验证: target_preds[i] 是对输入第 i 个 token 的下一步预测)
        accept_length = 0
        actual_gamma = len(draft_tokens)
        
        for i in range(actual_gamma):
            stats["position_attempts"][i] += 1
            if target_preds[i] == draft_tokens[i]:
                accept_length += 1
                stats["position_accepts"][i] += 1
            else:
                break
                
        stats["accepted_lengths"].append(accept_length)
        accepted_ids = draft_tokens[:accept_length]
        
        # 核心：获取 Bonus Token
        bonus_token = target_preds[accept_length]
        
        tokens_to_add = accepted_ids + [bonus_token]
        seq.extend(tokens_to_add)
        stats["new_tokens"] += len(tokens_to_add)

        # 核心：获取下一步草稿 , 比较巧妙 (推理时 batch_size 固定为 1)
        next_draft_tokens = torch.argmax(m_logits_stack[0, :gamma, accept_length, :], dim=-1).tolist()

        # ==========================================
        # 2. KV Cache 精确裁剪
        # ==========================================
        keep_len += (1 + accept_length)
        base_kv = trim_kv_cache(base_kv, keep_len)
        medusa_kv = trim_kv_cache(medusa_kv, keep_len)
        
        # next_token = bonus_token
        # draft_tokens = next_draft_tokens

        torch.cuda.synchronize()
        step_time = time.perf_counter() - t_step_start
        stats["forward_time"] += step_time
        stats["forward_count"] += 1

        # 打印 Debug
        if debug_log_file:
            truly_rejected_id = [draft_tokens[accept_length]] if accept_length < actual_gamma else []
            unseen_draft_ids = draft_tokens[accept_length + 1:] if accept_length + 1 < actual_gamma else []

            debug_log_file.write(f"Step {step_counter} | Accept Length: {accept_length}/{actual_gamma}\n")
            debug_log_file.write(f"\tForward Time: {step_time*1000:.2f} ms\n")
            debug_log_file.write(f"\t[Draft]   '{draft_tokens}' | {tokenizer.decode(draft_tokens)}\n")
            debug_log_file.write(f"\t[Accept]  '{accepted_ids}' | {tokenizer.decode(accepted_ids) if accepted_ids else '(None)'}\n")
            debug_log_file.write(f"\t[Reject]  '{truly_rejected_id}' | {tokenizer.decode(truly_rejected_id) if truly_rejected_id else '(None)'}\n")
            debug_log_file.write(f"\t[Bonus]   '{[bonus_token]}' | {tokenizer.decode([bonus_token])}\n")
            debug_log_file.write(f"\t[Unseen]  '{unseen_draft_ids}' | {tokenizer.decode(unseen_draft_ids) if unseen_draft_ids else '(None)'}\n")
            debug_log_file.write(f"\t[Final]   '{tokens_to_add}' | {tokenizer.decode(tokens_to_add)}\n")
            debug_log_file.write("-" * 80 + "\n")
            debug_log_file.flush()

        next_token = bonus_token
        draft_tokens = next_draft_tokens

    if debug_log_file:
        debug_log_file.write("\n" + "="*60 + "\n")
        debug_log_file.write("📄 [DEBUG] 最终完整生成的回答:\n")
        debug_log_file.write("="*60 + "\n")
        debug_log_file.write(tokenizer.decode(seq[prompt_len:]) + "\n")
        debug_log_file.write("="*60 + "\n\n")

    return stats


def run_benchmark(args):
    print("Loading AttnMedusa Model...")
    
    model = AttnMedusaModel.from_pretrained(
        medusa_head_name_or_path=args.medusa_model_path,
        base_model=args.base_model_path,
        medusa_num_heads=args.gamma
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    questions = []
    question_file = os.path.join(DATA_DIR, f"{args.bench_name}/question.jsonl")
    try:
        with open(question_file, "r") as f:
            for line in f: questions.append(json.loads(line))
    except FileNotFoundError:
        print(f"找不到数据集文件 {question_file}，使用内置 Dummy 数据测试...")
        questions = [{"turns": ["Please write a quicksort in python."]}] * args.num_samples

    # --- 1. 预热阶段 ---
    print("🔥 Warming up...")
    dummy_input = tokenizer(["Hello, who are you?"], return_tensors="pt").input_ids.to(model.base_model.device)
    benchmark_attn_medusa_generate(model, tokenizer, dummy_input, max_new_tokens=16, gamma=args.gamma)

    # --- 2. 纯粹的 Debug 阶段 ---
    if args.show_first_sample:
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(LOG_DIR, f"attn_medusa_benchmark_{timestamp}.log")
        print(f"\n📝 正在进行单样本详细解码追踪...")
        print(f"📂 日志文件: \033[96m{log_filename}\033[0m")
        
        q0 = questions[0]
        messages = [{"role": "user", "content": q0["turns"][0]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(model.base_model.device)
        
        with open(log_filename, "w", encoding="utf-8") as log_file:
            benchmark_attn_medusa_generate(model, tokenizer, input_ids, max_new_tokens=args.max_new_tokens, gamma=args.gamma, debug_log_file=log_file)
            
        print(f"✅ 追踪完成！\n")

    # --- 3. 正式的评测阶段 ---
    all_stats = []
    print("🚀 Starting benchmark...")
    
    for q in tqdm(questions[:args.num_samples], total=args.num_samples):
        messages = [{"role": "user", "content": q["turns"][0]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(model.base_model.device)
        
        start_w = time.perf_counter()
        stat = benchmark_attn_medusa_generate(model, tokenizer, input_ids, max_new_tokens=args.max_new_tokens, gamma=args.gamma)
        stat["total_wall_time"] = time.perf_counter() - start_w
        all_stats.append(stat)

    # --- 4. 结果统计 ---
    total_steps = sum(len(s["accepted_lengths"]) for s in all_stats)
    total_accepts = sum(sum(s["accepted_lengths"]) for s in all_stats)
    avg_accept_len = total_accepts / total_steps if total_steps > 0 else 0
    avg_forward_time = sum(s["forward_time"] for s in all_stats) / sum(s["forward_count"] for s in all_stats)
    
    total_tokens = sum(s["new_tokens"] for s in all_stats)
    total_time = sum(s["total_wall_time"] for s in all_stats)
    speed = total_tokens / total_time
    
    print(f"\n====== Attn-Medusa 测试结果 ======")
    print(f"Base Model:   {args.base_model_path}")
    print(f"Medusa Heads: {args.medusa_model_path}")
    print(f"1. 解码速度: {speed:.2f} tokens/s")
    print(f"2. 平均接受长度: {avg_accept_len:.2f}")
    print(f"3. 平均延时 (forward): {avg_forward_time*1000:.2f}ms | 生成总 Token 数: {total_tokens} | 总耗时: {total_time:.2f}s")
    
    pos_accepts_list = torch.stack([s["position_accepts"] for s in all_stats]).sum(dim=0).cpu().tolist()
    print("4. 逐位置接受率 (条件接受率 / Conditional Acceptance Rate):")
    
    current_denominator = total_steps
    for i, accepts in enumerate(pos_accepts_list):
        if i >= args.gamma: break
        if current_denominator > 0:
            rate = accepts / current_denominator
            print(f"   Token {i+1}: {rate*100:.2f}% ({int(accepts)}/{int(current_denominator)})")
        else:
            print(f"   Token {i+1}: 0.00% (0/0)")
        current_denominator = accepts

    # 打印内部组件测试结果
    avg_time_base = sum(s.get("time_base", 0) for s in all_stats) / sum(s.get("counts", 0) for s in all_stats)
    avg_time_decoder = sum(s.get("time_decoder", 0) for s in all_stats) / sum(s.get("counts", 0) for s in all_stats)
    avg_time_medusa = sum(s.get("time_medusa", 0) for s in all_stats) / sum(s.get("counts", 0) for s in all_stats)
    avg_internal_time = avg_time_base + avg_time_decoder + avg_time_medusa
    print(f"--- 内部组件时延分析 ---")
    if avg_internal_time > 0:
        print(f"   Base Model耗时:    {avg_time_base*1000:.2f}ms (占比 {avg_time_base/avg_internal_time*100:.1f}%)")
        print(f"   Decoder Layer耗时: {avg_time_decoder*1000:.2f}ms (占比 {avg_time_decoder/avg_internal_time*100:.1f}%)")
        print(f"   Medusa Heads耗时:  {avg_time_medusa*1000:.2f}ms (占比 {avg_time_medusa/avg_internal_time*100:.1f}%)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--medusa-model-path", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--bench-name", type=str, default="gsm8k")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--show-first-sample", action="store_true")
    args = parser.parse_args()
    
    run_benchmark(args)