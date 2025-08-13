from tqdm import tqdm
import json
from rouge import Rouge
import jieba
import argparse
import torch

from transformers import AutoTokenizer, AutoModel
rouge = Rouge()
model_path = "/home/lei/lei_data/yinyucup/models/Qwen3-8B" 
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
# model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")

# def get_pooled_embedding(text):
    # inputs = tokenizer(text, return_tensors="pt").to("cuda")
    # with torch.no_grad():
        # outputs = model(**inputs)
    # 平均池化（也可用 max pooling）
    # return outputs.last_hidden_state.mean(dim=1)


def calculate_rouge_l(candidate, reference):
    # 对中⽂⽂本进⾏分词处理
    candidate_tokens = ' '.join(jieba.cut(candidate))
    reference_tokens = ' '.join(jieba.cut(reference))
    if len(candidate_tokens) > 2048 or len(reference_tokens) > 2048:
        return {'f':0.0}
    scores = rouge.get_scores(candidate_tokens, reference_tokens)
    return scores[0]['rouge-l']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")
    parser.add_argument("--test_data", type=str, default="/data/lei_data/yinyucup/train_data/train.jsonl")
    parser.add_argument("--model_path", type=str, default="/data/lei_data/yinyucup/models/ATEC2025-Qwen-Base")
    
    parser.add_argument("--user_out_path", type=str, default="/home/lei/lei_data/yinyucup/result/train_predict.jsonl")
    
    parser.add_argument("--forget_out_path", type=str, default="/home/lei/lei_data/yinyucup/my-LLM-Unlearning/data/forget/forget_v5.jsonl")
    # choice 高于upper_threhold
    parser.add_argument("--retain_out_path", type=str, default="/home/lei/lei_data/yinyucup/my-LLM-Unlearning/data/retain/retain_v5.jsonl")
    
    args = parser.parse_args()
    test_data = args.test_data
    user_out_path = args.user_out_path
    
    # f_qa = open(args.qa_out_path, 'w', encoding='utf-8')
    # f_choice = open(args.choice_out_path, 'w', encoding='utf-8')
    # f_nonc = open(args.nonc_out_path, 'w', encoding='utf-8')
    
    f_forget = open(args.forget_out_path, 'w', encoding='utf-8')
    f_retain = open(args.retain_out_path, 'w', encoding='utf-8')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code = True)
    f_rouge = open("/home/lei/lei_data/yinyucup/rouge.txt", "w")
    f_token = open("/home/lei/lei_data/yinyucup/token_len.txt", "w")
    dataset = []
    original_line = []
    with open(test_data,'r',encoding='utf-8') as f:
        print("loading dataset...")
        for line in f:
            data = json.loads(line)
            dataset.append(data)
            original_line.append(line)
    # print(dataset[0])
    
    ###########选⼿模型产⽣的输出###########
    data_qwen_gen = []
    with open(user_out_path,'r',encoding='utf-8') as f:
        print("loading user model output...")
        for line in f:
            data = json.loads(line)
            data_qwen_gen.append(data['text'])
    
    num_f = 0
    num_r = 0
    token_len_list = []
    for i in tqdm(range(len(dataset))):
        question = dataset[i]['conversations'][0]['value']
        answer = dataset[i]['conversations'][1]['value']
        messages = [
            {"role": "user", "content": question},
            # {"role": "assistant", "content": entry["conversations"][1]["value"]}
        ]
        tokenized_output = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=False,
            return_tensors="pt",
        )
        token_len = tokenized_output.shape[1]
        # x = 0
        # y = 0
        
        if i < 30000 or i >= 40000:
            candidate_privacy = dataset[i]['conversations'][1]['value']
            reference_privacy = data_qwen_gen[i]
            if reference_privacy == "":
                print(i)
            try:
                rouge_l_score_privacy = calculate_rouge_l(candidate_privacy, reference_privacy)['f']
                f_rouge.write(f"{rouge_l_score_privacy}\n")
                # embd_candidate = get_pooled_embedding(dataset[i]['conversations'][1]['value'])
                # embd_reference = get_pooled_embedding(data_qwen_gen[i])
                # cos_sim = torch.cosine_similarity(embd_candidate, embd_reference, dim=1).item()
                # print(rouge_l_score_privacy, cos_sim)
                f_token.write(f"{token_len}\n")
            except:
                print(i)
                f_rouge.write(f"0.0\n")
                f_token.write(f"{token_len}\n")
                continue
            
            if rouge_l_score_privacy >= 0.5:
                if token_len >= 15:
                    num_f += 1
                f_forget.write(original_line[i])
                    
            if rouge_l_score_privacy < 0.25:
                if token_len <= 72:
                    num_r += 1
                f_retain.write(original_line[i])
                    
        max_choice = 5000
        if i >= 30000 and i < 40000:
            candidate_privacy = dataset[i]['conversations'][1]['value']
            reference_privacy = data_qwen_gen[i]
            if reference_privacy == "":
                print(i)
            try:
                rouge_l_score_privacy = calculate_rouge_l(candidate_privacy, reference_privacy)['f']
                f_rouge.write(f"{rouge_l_score_privacy}\n")
                f_token.write(f"{token_len}\n")
            except:
                print(i)
                f_rouge.write(f"0.0\n")
                f_token.write(f"{token_len}\n")
                continue
            if rouge_l_score_privacy <= 0.8:
                max_choice = max_choice - 1
                f_retain.write(original_line[i])
            if max_choice == 0:
                break

    print(f"forget:{num_f}")
    print(f"retain:{num_r}")
    