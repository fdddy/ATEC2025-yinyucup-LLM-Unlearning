from tqdm import tqdm
import json
from rouge import Rouge
import jieba
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

rouge = Rouge()
def calculate_rouge_l(candidate, reference):
    # 对中⽂⽂本进⾏分词处理
    candidate_tokens = ' '.join(jieba.cut(candidate))
    reference_tokens = ' '.join(jieba.cut(reference))
    if len(candidate_tokens) > 2048 or len(reference_tokens) > 2048:
        return {'f':0.0}
    scores = rouge.get_scores(candidate_tokens, reference_tokens)
    return scores[0]['rouge-l']

# 平均池化，忽略 padding 部分
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # [batch, seq_len, hidden]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)
    return sum_embeddings / sum_mask

# 获取句子向量
def get_sentence_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = mean_pooling(outputs, inputs['attention_mask'])  # [1, hidden_size]
    return embedding

if __name__ == "__main__":

    # 加载优化过的中文BERT模型
    model_name = "/home/lei/lei_data/yinyucup/models/text2vec-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().cuda()  # 使用 GPU

    forgeted_nonpriv_path = "./data/process/forgeted_nonpriv.jsonl"
    non_forget_priv_path = "./data/process/non_forget_priv.jsonl"

    forget_path = "./data/forget/forget.jsonl"
    retain_path = "./data/retain/retain.jsonl"

    # 加载 forget/retain 集合（作为语义比对目标集）
    def load_jsonl(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    forget_data = load_jsonl(forget_path)
    retain_data = load_jsonl(retain_path)

    data_forgeted_nonpriv = load_jsonl(forgeted_nonpriv_path)
    data_nonforget_privacy = load_jsonl(non_forget_priv_path)

    # 获取 ques 列表并嵌入
    def get_embeddings(data_list):
        texts = [entry['conversations'][1]['value'] for entry in data_list]
        with torch.no_grad():
            return [get_sentence_embedding(model, tokenizer, text) for text in texts]

    # 已知的两个错误划分集合
    # data_forgeted_nonpriv: 错误归为 forget 的样本（应和 forget 对比）
    # data_nonforget_privacy: 错误归为 retain 的样本（应和 retain 对比）

    # 提前嵌入比对集合
    forget_embeddings = get_embeddings(forget_data)
    retain_embeddings = get_embeddings(retain_data)

    # 设置相似度阈值（可根据实验调整）
    SIM_THRESHOLD = 0.7

    # 最终清洗后的集合
    cleaned_forget = forget_data.copy()
    cleaned_retain = retain_data.copy()
    # i = 0
    # # 清洗 data_forgeted_nonpriv：比对 forget，剔除相似度高的条目，加到 retain
    # for item in tqdm(data_forgeted_nonpriv):
    #     emb_item = get_sentence_embedding(model, tokenizer, item['conversations'][0]['value'])
        
    #     # 计算当前条目与 forget 集合所有条目的相似度
    #     for emb_f, ref_item in tqdm(zip(forget_embeddings, forget_data)):

    #         rouge_l_score = calculate_rouge_l(item['conversations'][0]['value'], ref_item['conversations'][0]['value'])['f']
    #         max_sim = F.cosine_similarity(emb_item, emb_f).item()
    #         # print(f"rouge_l:{rouge_l_score}")
    #         if max_sim > SIM_THRESHOLD and rouge_l_score >= 0.8:
    #             i += 1
    #             # 如果相似度高，说明该条目应从 forget 移至 retain
    #             if ref_item in cleaned_forget:
    #                 cleaned_forget.remove(ref_item)  # 移除原本的 forget
    #                 cleaned_retain.append(ref_item)  # 加入 retain
    #             break
    j = 0
    # 清洗 data_nonforget_privacy：比对 retain，剔除相似度高的条目，加到 forget
    for item in tqdm(data_nonforget_privacy):
        emb_item = get_sentence_embedding(model, tokenizer, item['conversations'][1]['value'])

        # 计算当前条目与 retain 集合所有条目的相似度
        # 只根据问题进行判断是否有局限性
        for emb_r, ref_item in zip(retain_embeddings, retain_data):
            rouge_l_score = calculate_rouge_l(item['conversations'][1]['value'], 
                                ref_item['conversations'][1]['value'])['f']
            max_sim = F.cosine_similarity(emb_item, emb_r).item()
            # print(f"cos_sim:{max_sim}")
            if max_sim > SIM_THRESHOLD or rouge_l_score >= 0.5:
                j += 1
                # 如果相似度高，说明该条目应从 retain 移至 forge   
                if ref_item in cleaned_retain:
                    cleaned_retain.remove(ref_item)  # 移除原本的 retain
                    cleaned_forget.append(ref_item)  # 加入 forget
                break

    # 保存结果（可选）
    def save_jsonl(path, data_list):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    save_jsonl('./data/forget/cleaned_forget.jsonl', cleaned_forget)
    save_jsonl('./data/retain/cleaned_retain.jsonl', cleaned_retain)

    # print(i)
    print(j)
    
    print(f"raw forget len:{len(forget_data)}")
    print(f"raw retain len:{len(retain_data)}")

    print(f"cleaned_forget len:{len(cleaned_forget)}")
    print(f"cleaned_retain len:{len(cleaned_retain)}")