import os
import torch
import argparse
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_lora_adapter(base_model_path: str, adapter_path: str, output_dir: str):
    """
    合并 LoRA 适配器到基础模型
    
    Args:
        base_model_path: 基础模型路径
        adapter_path: LoRA 适配器路径
        output_dir: 输出目录
    """
    print(f"Base model: {base_model_path}")
    print(f"LoRA adapter: {adapter_path}")
    print(f"Output directory: {output_dir}")
    
    # 验证路径
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    
    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Loading base model...")
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Loading tokenizer...")
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading LoRA adapter...")
    # 加载 PEFT 模型（带适配器）
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging adapter weights...")
    # 合并适配器权重
    merged_model = model.merge_and_unload()
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Saving merged model to: {output_dir}")
    # 保存合并后的模型
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # 保存 tokenizer
    try:
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Warning: Failed to save tokenizer: {e}")
        # 创建干净的 tokenizer
        clean_tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        clean_tokenizer.save_pretrained(output_dir)
        print("Saved tokenizer with cleaned configuration")
    
    print("Merge completed successfully!")
    
    # 清理内存
    del merged_model, model, base_model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, default = '/home/lei/lei_data/yinyucup/models/ATEC2025-Qwen-Base',
                       help="Path to the base model")
    parser.add_argument("--adapter", type=str, default = './adapter/ME+GD_epoch5_3e-05_maskFalse_0.9_v5/final_lora_model',
                       help="Path to the LoRA adapter")
    parser.add_argument("--output", type=str, default = './merged_model/ME+GD_epoch3_3e-05_maskFalse_0.9_v5',
                       help="Output directory for merged model")
    
    args = parser.parse_args()
    
    # 检查输出目录是否存在
    if os.path.exists(args.output):
        response = input(f"Output directory {args.output} already exists. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    merge_lora_adapter(args.base_model, args.adapter, args.output)

if __name__ == "__main__":
    main()

