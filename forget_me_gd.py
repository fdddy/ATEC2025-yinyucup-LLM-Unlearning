import os
import shutil
import warnings
from pathlib import Path

import json
import torch
import swanlab
import argparse
import datasets
from datetime import datetime
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from dataset import SimpleTextForgetDatasetQA, simple_data_collator_forget
from dataset import IDKTextForgetDatasetQA, idk_data_collator_forget
from trainer.losses import get_loss
from utils import set_random_seed

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def test_model_gradient_flow(model, batch):
    """测试模型的梯度流"""
    forget_inputs = batch[0]
    input_ids, labels, attention_mask = forget_inputs
    
    print("=== Gradient Flow Test ===")
    
    # 1. 检查模型状态
    model.train()
    print(f"Model training: {model.training}")
    
    # 2. 检查参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {len(trainable_params)}")
    
    # 3. 强制开启梯度
    with torch.enable_grad():
        # 4. 前向传播
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        print(f"Logits requires_grad: {logits.requires_grad}")
        print(f"Logits shape: {logits.shape}")
        
        # 5. 简单损失测试
        simple_loss = logits.sum()  # 最简单的损失
        print(f"Simple loss requires_grad: {simple_loss.requires_grad}")
        
        return simple_loss

'''
def train_model_accelerate(args):
    # 1. 初始化Accelerator（按官方文档）
    accelerator = Accelerator()
    
    # 打印分布式信息
    if accelerator.is_main_process:
        print(f"Distributed type: {accelerator.distributed_type}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Local process index: {accelerator.local_process_index}")
        print(f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")

    # 创建日志目录
    log_dir = "/home/lei/lei_data/yinyucup/my-LLM-Unlearning/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    f_log = open(log_file, "a") if accelerator.is_main_process else None
    
    model_path = "/home/lei/lei_data/yinyucup/models/ATEC2025-Qwen-Base"
    
    # 2. 在main_process_first中加载tokenizer（按官方文档）
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 在main_process_first中加载模型（按官方文档）
    if accelerator.is_main_process:
        print("Loading base model...")

    with accelerator.main_process_first():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    # 4. 配置LoRA（按官方文档）
    if accelerator.is_main_process:
        print("Configuring LoRA...")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=getattr(args, 'lora_r', 16),
        lora_alpha=getattr(args, 'lora_alpha', 32),
        lora_dropout=getattr(args, 'lora_dropout', 0.1),
        target_modules=getattr(args, 'lora_target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias="none",
    )
    
    # 5. 应用PEFT（按官方文档）
    if accelerator.is_main_process:
        print("Applying LoRA to model...")
    model = get_peft_model(model, peft_config)
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # 解析动态路径
    curr_save_dir = args.output_dir
    if isinstance(curr_save_dir, str) and '${' in curr_save_dir:
        def replace_variables(template, args):
            result = template
            for attr_name in dir(args):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(args, attr_name)
                        if not callable(attr_value):
                            placeholder = f'${{{attr_name}}}'
                            if placeholder in result:
                                result = result.replace(placeholder, str(attr_value))
                    except:
                        continue
            return result
        
        curr_save_dir = replace_variables(curr_save_dir, args)

    # 6. 数据加载（使用main_process_first）
    if accelerator.is_main_process:
        print("Loading data...")

    with accelerator.main_process_first():
        forget_data = datasets.load_dataset('json', data_files=args.forget_data_path, split='train')
        retain_data = datasets.load_dataset('json', data_files=args.retain_data_path, split='train')

        torch_format_dataset = SimpleTextForgetDatasetQA(tokenizer=tokenizer,
                                                        model_family=args.model_family,
                                                        forget_data=forget_data,
                                                        retain_data=retain_data,
                                                        max_length=args.max_length,
                                                        mask=args.mask)

    train_loader = DataLoader(torch_format_dataset, batch_size=args.bs, shuffle=True, 
                             collate_fn=simple_data_collator_forget)

    print(f"Dataset size: {len(torch_format_dataset)}")
    print(f"Forget data size: {len(forget_data)}, Retain data size: {len(retain_data)}")

    # 验证数据格式
    batch = next(iter(train_loader))
    print(f"Batch format: {len(batch)} types")
    print(f"Forget batch shape: {batch[0][0].shape}")
    print(f"Retain batch shape: {batch[1][0].shape}")

    # 7. 优化器设置
    if accelerator.is_main_process:
        print("Setting up optimizer for LoRA parameters...")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found! Check LoRA configuration.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)

    total_steps = len(train_loader) * args.num_epochs
    actual_total_steps = total_steps // accelerator.gradient_accumulation_steps
    num_warmup_steps = int(args.warmup_ratio * actual_total_steps)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=actual_total_steps,
    )

    # 8. 使用accelerator.prepare（按官方文档）
    model, train_loader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_loader, optimizer, lr_scheduler
    )

    # 9. 检查是否使用ZeRO Stage 3（按官方文档）
    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
        actual_zero_stage = accelerator.state.deepspeed_plugin.zero_stage
        if accelerator.is_main_process:
            # 不兼容，自动降级到 Zero Stage 2 
            print(f"*** DeepSpeed ZeRO Stage: {actual_zero_stage} ***")
            print(f"*** Using ZeRO-3: {is_ds_zero_3} ***")
    
    global_step = 0
    print(f"Starting training for {args.num_epochs} epochs...")

    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        swanlab.init(
            project="yinyucup-llm-unlearning",
            config=vars(args),
            name=f"{args.loss_type}_forget{args.forget_coeff}_retain{args.regularization_coeff}_{timestamp}",
        )

    # 10. 训练循环（按官方文档，ZeRO-3可以使用accumulate）
    for ep in range(args.num_epochs):
        model.train()
        tr_loss = []
        forget_losses = []
        reg_losses = []
        
        for batch in tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {ep}"):
            # ZeRO Stage 3可以安全使用accumulate
            with accelerator.accumulate(model):
                # 计算损失
                forget_loss, regularization_loss = get_loss(
                    model=model, 
                    ref_model=None,
                    inputs=batch, 
                    loss_type=args.loss_type, 
                    beta=0.1
                )
                
                # 组合总损失
                total_loss = args.forget_coeff * forget_loss + args.regularization_coeff * regularization_loss
                
                # 使用accelerator.backward（按官方文档）
                accelerator.backward(total_loss)

                # 梯度裁剪
                if accelerator.sync_gradients:
                    if hasattr(args, 'max_grad_norm') and args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 记录损失（只在主进程）
            if accelerator.is_main_process:
                tr_loss.append(total_loss.item())
                forget_losses.append(forget_loss.item())
                if isinstance(regularization_loss, torch.Tensor):
                    reg_losses.append(regularization_loss.item())
                else:
                    reg_losses.append(regularization_loss)
            
            # 更新步数
            if accelerator.sync_gradients:
                global_step += 1

                # 日志记录
                if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                    recent_steps = min(args.logging_steps, len(tr_loss))
                    avg_loss = sum(tr_loss[-recent_steps:]) / recent_steps
                    avg_forget_loss = sum(forget_losses[-recent_steps:]) / recent_steps
                    avg_reg_loss = sum(reg_losses[-recent_steps:]) / recent_steps
                    
                    log_msg = f"Epoch: {ep}, Step: {global_step}, Total Loss: {avg_loss:.6f}, " \
                            f"Forget Loss: {avg_forget_loss:.6f}, Reg Loss: {avg_reg_loss:.6f}, " \
                            f"LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                    
                    swanlab.log({
                        "step": global_step,
                        "epoch": ep,
                        "total_loss": avg_loss,
                        "forget_loss": avg_forget_loss,
                        "regularization_loss": avg_reg_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                    })
                    print(log_msg)
                    if f_log:
                        f_log.write(log_msg + "\n")
                        f_log.flush()

        # Epoch结束处理
        if accelerator.is_main_process and tr_loss:
            epoch_avg_loss = sum(tr_loss) / len(tr_loss)
            epoch_avg_forget = sum(forget_losses) / len(forget_losses)
            epoch_avg_reg = sum(reg_losses) / len(reg_losses)
            
            epoch_log = f"Epoch {ep} completed - Avg Total Loss: {epoch_avg_loss:.6f}, " \
                       f"Avg Forget Loss: {epoch_avg_forget:.6f}, Avg Reg Loss: {epoch_avg_reg:.6f}"
            
            print(epoch_log)
            if f_log:
                f_log.write(epoch_log + "\n")
                f_log.flush()

        # 等待所有进程完成当前 epoch
        accelerator.wait_for_everyone()
        
        # 保存检查点
        if hasattr(args, 'save_steps') and (ep + 1) % args.save_steps == 0:
            if accelerator.is_main_process:
                save_path = f"{curr_save_dir}/checkpoint-epoch-{ep+1}"
                os.makedirs(save_path, exist_ok=True)
                
                # 等待所有进程
                accelerator.wait_for_everyone()
                
                # 获取unwrapped模型
                unwrapped_model = accelerator.unwrap_model(model)
                
                # 如果是ZeRO-3，使用特殊保存方法
                if is_ds_zero_3:
                    # 使用accelerator的save方法
                    accelerator.save_model(unwrapped_model, save_path)
                else:
                    # 常规保存
                    unwrapped_model.save_pretrained(save_path)
                
                # 保存tokenizer
                try:
                    tokenizer.save_pretrained(save_path)
                except TypeError as e:
                    if "dtype" in str(e) or "not JSON serializable" in str(e):
                        clean_tokenizer = AutoTokenizer.from_pretrained(
                            tokenizer.name_or_path,
                            trust_remote_code=True
                        )
                        clean_tokenizer.save_pretrained(save_path)
                        print("Saved checkpoint tokenizer with cleaned configuration")
                    else:
                        raise e
                print(f"ZeRO-3 checkpoint saved to {save_path}")
    
    # 等待所有进程完成训练
    accelerator.wait_for_everyone()
    
    # 最终模型保存
    if accelerator.is_main_process:
        final_save_path = f"{curr_save_dir}/final_lora_model"
        os.makedirs(final_save_path, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        
        # 如果是ZeRO-3，使用特殊保存方法
        if is_ds_zero_3:
            accelerator.save_model(unwrapped_model, final_save_path)
        else:
            unwrapped_model.save_pretrained(final_save_path)
        
        try:
            tokenizer.save_pretrained(final_save_path)
        except TypeError as e:
            if "dtype" in str(e) or "not JSON serializable" in str(e):
                clean_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer.name_or_path,
                    trust_remote_code=True
                )
                clean_tokenizer.save_pretrained(final_save_path)
                print("Saved tokenizer with cleaned configuration")
            else:
                raise e

        print(f"Final model saved to {final_save_path}")
        swanlab.finish()
    
    if f_log:
        f_log.close()
    
    # 清理内存
    del model
    del optimizer
    del lr_scheduler
    del train_loader
    del tokenizer 
    
    accelerator.free_memory()
    del accelerator
    
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
'''

def train_model_accelerate(args):

    seed = args.seed
    set_random_seed(seed)
    # 检查是否使用配置文件
    if hasattr(args, 'config_file') and args.config_file:
        print(f"Using accelerate config file: {args.config_file}")
    
    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    accelerator = Accelerator(mixed_precision='bf16')
    
    # 打印分布式信息
    if accelerator.is_main_process:
        print(f"Distributed type: {accelerator.distributed_type}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Local process index: {accelerator.local_process_index}")
        print(f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")

    # 创建日志目录
    log_dir = "/home/lei/lei_data/yinyucup/my-LLM-Unlearning/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    f_log = open(log_file, "a") if accelerator.is_main_process else None
    

    model_path = "/home/lei/lei_data/yinyucup/models/ATEC2025-Qwen-Base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型 - DeepSpeed 兼容
    if accelerator.is_main_process:
        print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    # 配置 LoRA - 针对 DeepSpeed 优化
    if accelerator.is_main_process:
        print("Configuring LoRA...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 因果语言模型
        inference_mode=False,  # 训练模式
        r=getattr(args, 'lora_r', 16),  # LoRA rank，默认16
        lora_alpha=getattr(args, 'lora_alpha', 32),  # LoRA scaling parameter，默认32
        lora_dropout=getattr(args, 'lora_dropout', 0.1),  # LoRA dropout，默认0.1
        # 目标模块 - 根据模型架构调整
        target_modules=getattr(args, 'lora_target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention layers
            "gate_proj", "up_proj", "down_proj"  # MLP layers for Qwen/LLaMA style models
        ]),
        bias="none",  # 不训练bias
        modules_to_save=None,  # 不保存额外模块
    )
    
    # 应用 LoRA 到模型
    if accelerator.is_main_process:
        print("Applying LoRA to model...")
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数信息
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # 解析动态路径
    curr_save_dir = args.output_dir
    if isinstance(curr_save_dir, str) and '${' in curr_save_dir:
        # 通用的变量替换函数
        def replace_variables(template, args):
            """替换模板中的变量"""
            result = template
            # 获取配置对象的所有属性
            for attr_name in dir(args):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(args, attr_name)
                        if not callable(attr_value):
                            placeholder = f'${{{attr_name}}}'
                            if placeholder in result:
                                result = result.replace(placeholder, str(attr_value))
                    except:
                        continue
            return result
        
        curr_save_dir = replace_variables(curr_save_dir, args)

    if accelerator.is_main_process:
        print("Loading data...")

    forget_data = datasets.load_dataset('json', data_files=args.forget_data_path,
                                        split='train')
    retain_data = datasets.load_dataset('json', data_files=args.retain_data_path,
                                        split='train')
    
    if args.loss_type == "ME+GD":
        # 每个样本返回: [forget_sample, retain_sample]
        # 其中每个sample是: (input_ids, labels, attention_mask) # shape: [batch_size, seq_len]
        torch_format_dataset = SimpleTextForgetDatasetQA(tokenizer=tokenizer,
                                                        model_family=args.model_family,
                                                        forget_data=forget_data,
                                                        retain_data=retain_data,
                                                        max_length=args.max_length,
                                                        mask=args.mask)
        train_loader = DataLoader(torch_format_dataset, batch_size=args.bs, shuffle=True, collate_fn=simple_data_collator_forget)
    elif args.loss_type == "IDK+AP":
        torch_format_dataset = IDKTextForgetDatasetQA(tokenizer=tokenizer,
                                                        model_family=args.model_family,
                                                        forget_data=forget_data,
                                                        retain_data=retain_data,
                                                        max_length=args.max_length,
                                                        mask=args.mask)
        train_loader = DataLoader(torch_format_dataset, batch_size=args.bs, shuffle=True, collate_fn=idk_data_collator_forget)
    
    if accelerator.is_main_process:
        print(f"Dataset size: {len(torch_format_dataset)}")
        print(f"Forget data size: {len(forget_data)}, Retain data size: {len(retain_data)}")# 在训练开始前验证数据格式
        batch = next(iter(train_loader))
        print(f"Batch format: {len(batch)} types")
        print(f"Forget batch shape: {batch[0][0].shape}")  # input_ids shape
        print(f"Retain batch shape: {batch[1][0].shape}")  # input_ids shape

    # 只优化 LoRA 参数
    if accelerator.is_main_process:
        print("Setting up optimizer for LoRA parameters...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    # 检查是否有可训练参数
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found! Check LoRA configuration.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)

    total_steps = len(train_loader) * args.num_epochs
    # DeepSpeed 会处理梯度累积，所以实际步数需要调整
    actual_total_steps = total_steps // accelerator.gradient_accumulation_steps
    num_warmup_steps = int(args.warmup_ratio * actual_total_steps)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=actual_total_steps,
    )
    model, optimizer, lr_scheduler, train_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader
    )

    if accelerator.is_main_process:
        print(f"Starting training for {args.num_epochs} epochs...")

        timestamp = datetime.now().strftime("%m%d_%H%M")
        swanlab.init(
            project="yinyucup-llm-unlearning",  # 项目名称
            config=vars(args),         # 保存训练超参
            name=f"{args.loss_type}_epoch{args.num_epochs}_lr{args.lr}_maxlen{args.max_length}_forget{args.forget_coeff}_retain{args.regularization_coeff}_{timestamp}",  # 任务名
        )
        
    global_step = 0
    for ep in range(args.num_epochs):
        model.train()
        tr_loss, forget_losses, reg_losses = [], [], []

        # 用于跟踪梯度累积
        accumulated_loss = 0.0
        accumulated_forget_loss = 0.0
        accumulated_reg_loss = 0.0
        accumulation_count = 0
        
        for step, batch in enumerate(tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {ep}")):
            # 计算自定义损失函数
            forget_loss, regularization_loss = get_loss(
                model=model, 
                ref_model=None,
                inputs=batch, 
                loss_type=args.loss_type, 
                beta=args.beta
            )
            total_loss = args.forget_coeff * forget_loss + args.regularization_coeff * regularization_loss
            scaled_loss = total_loss / accelerator.gradient_accumulation_steps

            accelerator.backward(scaled_loss)

            # 累积损失用于记录
            accumulated_loss += total_loss.item()
            accumulated_forget_loss += forget_loss.item()
            if isinstance(regularization_loss, torch.Tensor):
                accumulated_reg_loss += regularization_loss.item()
            else:
                accumulated_reg_loss += regularization_loss
            accumulation_count += 1
                        
            if (step + 1) % accelerator.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                # 梯度裁剪
                if hasattr(args, 'max_grad_norm') and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if accelerator.is_main_process:
                    avg_loss = accumulated_loss / accumulation_count
                    avg_forget_loss = accumulated_forget_loss / accumulation_count
                    avg_reg_loss = accumulated_reg_loss / accumulation_count
                    
                    tr_loss.append(avg_loss)
                    forget_losses.append(avg_forget_loss)
                    reg_losses.append(avg_reg_loss)

                # 重置累积变量
                accumulated_loss = accumulated_forget_loss = accumulated_reg_loss = 0.0
                accumulation_count = 0
                global_step += 1

                # 日志记录
                if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                    recent_steps = min(args.logging_steps, len(tr_loss))
                    avg_loss = sum(tr_loss[-recent_steps:]) / recent_steps
                    avg_forget_loss = sum(forget_losses[-recent_steps:]) / recent_steps
                    avg_reg_loss = sum(reg_losses[-recent_steps:]) / recent_steps
                    
                    log_msg = f"Epoch: {ep}, Step: {global_step}, Total Loss: {avg_loss:.6f}, " \
                            f"Forget Loss: {avg_forget_loss:.6f}, Reg Loss: {avg_reg_loss:.6f}, " \
                            f"LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                    
                    swanlab.log({
                        "step": global_step,
                        "epoch": ep,
                        "total_loss": avg_loss,
                        "forget_loss": avg_forget_loss,
                        "regularization_loss": avg_reg_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                    })
                    print(log_msg)
                    if 'f_log' in locals():
                        f_log.write(log_msg + "\n")
                        f_log.flush()

        # Epoch结束时的处理
        if accelerator.is_main_process and tr_loss:
            epoch_avg_loss = sum(tr_loss) / len(tr_loss)
            epoch_avg_forget = sum(forget_losses) / len(forget_losses)
            epoch_avg_reg = sum(reg_losses) / len(reg_losses)
            
            epoch_log = f"Epoch {ep} completed - Avg Total Loss: {epoch_avg_loss:.6f}, " \
                       f"Avg Forget Loss: {epoch_avg_forget:.6f}, Avg Reg Loss: {epoch_avg_reg:.6f}"
            
            print(epoch_log)
            if 'f_log' in locals():
                f_log.write(epoch_log + "\n")
                f_log.flush()

        # 等待所有进程完成当前 epoch
        accelerator.wait_for_everyone()
        
        # 保存 LoRA 检查点
        if hasattr(args, 'save_steps') and (ep + 1) % args.save_steps == 0:
            if accelerator.is_main_process:
                # 获取原始 PEFT 模型
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = f"{curr_save_dir}/checkpoint-epoch-{ep}"
                os.makedirs(save_path, exist_ok=True)
                
                # 保存 LoRA 适配器
                unwrapped_model.save_pretrained(save_path)
                
                # 清理 tokenizer 配置并保存
                try:
                    # 尝试直接保存
                    tokenizer.save_pretrained(save_path)
                except TypeError as e:
                    if "dtype" in str(e) or "not JSON serializable" in str(e):
                        # 如果遇到序列化问题，创建干净的 tokenizer
                        clean_tokenizer = AutoTokenizer.from_pretrained(
                            tokenizer.name_or_path,
                            trust_remote_code=True
                        )
                        clean_tokenizer.save_pretrained(save_path)
                        print("Saved checkpoint tokenizer with cleaned configuration")
                    else:
                        raise e
                print(f"LoRA checkpoint saved to {save_path}")
    
    # 等待所有进程完成训练
    accelerator.wait_for_everyone()
    
    # 训练结束，保存最终的 LoRA 模型
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        final_save_path = f"{curr_save_dir}/final_lora_model"
        os.makedirs(final_save_path, exist_ok=True)
        
        unwrapped_model.save_pretrained(final_save_path)
        
        try:
            tokenizer.save_pretrained(final_save_path)
        except TypeError as e:
            if "dtype" in str(e) or "not JSON serializable" in str(e):
                # 如果遇到序列化问题，创建干净的 tokenizer
                clean_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer.name_or_path,
                    trust_remote_code=True
                )
                clean_tokenizer.save_pretrained(final_save_path)
                print("Saved tokenizer with cleaned configuration")
            else:
                raise e

        print(f"Final LoRA model saved to {final_save_path}")
        swanlab.finish()

    if accelerator.is_main_process:
        f_log.close()
    del model
    del optimizer
    del lr_scheduler
    del train_loader
    del tokenizer 

    accelerator.free_memory()
    del accelerator
    
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print(f"Process {os.getpid()} starting...")
    print(f"Local rank: {os.environ.get('LOCAL_RANK', 'N/A')}")
    print(f"World size: {os.environ.get('WORLD_SIZE', 'N/A')}")

    parser = argparse.ArgumentParser(description='LLM Unleaning, Forget loss: ME(Max Entropy), regress loss: GD')
    parser.add_argument('--config_file', type=str, help='Path to accelerate config file')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs to train')
    parser.add_argument('--bs', type=int, default=1, help='batch size') 
    # for ME+GD loss, max_length = 400 is ok; for IDK + AP loss(3 types), max_length need to reduce
    # max_length 设置过小，问题存在被截断的可能
    parser.add_argument('--max_length', type=int, default=400, help='max length of the input sequence')

    parser.add_argument('--model_path', type=str, default='/home/lei/lei_data/yinyucup/models/ATEC2025-Qwen-Base')
    parser.add_argument('--model_family', type=str, default='qwen2.5-8b')
    parser.add_argument('--loss_type', type=str, default='ME+GD')
    # parser.add_argument('--loss_type', type=str, default='IDK+AP')

    parser.add_argument('--forget_data_path', type=str, default='./data/forget/forget_v5.jsonl')
    parser.add_argument('--retain_data_path', type=str, default='./data/retain/retain_v5.jsonl')
    parser.add_argument('--mask', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1001)

    # 多卡并行时，实际 batch增大，需要调大学习率
    #  lr需根据总step数量动态调整，当训练数据过少时，需增大学习率保证原收敛效果
    parser.add_argument('--lr', type=float, default=3e-5, help='Initial learning rate for the optimizer.')
    parser.add_argument('--wd', type=float, default=0.01, help='Initial weigth_decay for the optimizer.')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                        choices=['linear', 'cosine', 'constant', 'constant_with_warmup'], 
                        help='Type of learning rate scheduler.')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Ratio of total training steps for warmup.')
    parser.add_argument('--logging_steps', type=int, default=1)
    parser.add_argument('--save_steps', type=int, default=10)

    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    parser.add_argument('--forget_coeff', type=float, default=0.9)
    parser.add_argument('--regularization_coeff', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)

    parser.add_argument('--save_root', type=str, default='adapter')
    parser.add_argument('--output_dir', type=str, default='${save_root}/${loss_type}_epoch${num_epochs}_${lr}_mask${mask}_${forget_coeff}_v5')

    args = parser.parse_args()

    import time
    t = time.time()
    train_model_accelerate(args)
    if Accelerator().is_main_process:
        print(f"Accelerate training time for this trial: {time.time() - t:.4f}s")

    
