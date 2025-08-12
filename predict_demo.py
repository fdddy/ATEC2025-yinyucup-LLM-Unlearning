import os
from typing import List
import pdb
import json
import sys
from tqdm import tqdm
import argparse
import logging

# 固定写死 官网才能看到相关日志
log_file_path = './user.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

result = []

def infer_batch(engine: 'InferEngine', infer_requests: List['InferRequest']):
    logging.info("dataset split success, now infering......")
    request_config = RequestConfig(max_tokens=2048, temperature=0.0)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])

    for index, response in enumerate(resp_list):
        dict = {}
        res = resp_list[index].choices[0].message.content
        logging.info(f"lm response: {res}")
        # 若输出为空，改输出为 " "
        if res == "":
            res = " "
        dict['text'] = res
        result.append(dict)

if __name__ == '__main__':
    try:
        logging.info("success in to predict script, now loading user model......")
        parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")
        parser.add_argument("--model_path", type=str, default="./merged_model/ME+GD_epoch3_3e-05_maskFalse_0.9_v5")
        parser.add_argument("--data_path", type=str, default="./data/valid.jsonl")
        parser.add_argument("--output_path", type=str, default="./results/ME+GD_epoch3_3e-05_maskFalse_0.9_v5.jsonl")
        parser.add_argument("--model_type", type=str, default="qwen2_5")
        # parser.add_argument("--model_path", type=str, default="./models/Qwen3-8B")
        # parser.add_argument("--data_path", type=str, default="./train_data/valid.jsonl")
        # parser.add_argument("--output_path", type=str, default="./result/predictions_qwen3.jsonl")
        # parser.add_argument("--model_type", type=str, default="qwen3")
        parser.add_argument("--tensor_parallel_size", type=int, default=4)

        args = parser.parse_args()
        from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
        from swift.plugin import InferStats
        from swift.llm import VllmEngine

        model_path = args.model_path
        model_type = args.model_type
        output_path = args.output_path
        tensor_parallel_size = args.tensor_parallel_size

        model = model_path
        infer_backend = 'vllm'

        logging.info(f"param model path: {model_path}")
        logging.info(f"param outputpath: {output_path}")
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        
        if infer_backend == 'pt':
            engine = PtEngine(model, model_type=model_type, max_batch_size=64)
        # vllm引擎
        elif infer_backend == 'vllm':
            engine = VllmEngine(
                model, 
                model_type=model_type, 
                gpu_memory_utilization=0.8,
                tensor_parallel_size=tensor_parallel_size
            )
            
        logging.info("user model load success now begin split dataset")
        dataset = []
        # 逐行读取
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line))

        res = []
        for idx, data in tqdm(enumerate(dataset)):
            input = data['conversations'][0]['value']
            data_new = {}
            data_new['messages'] = []
            dict = {}
            dict['role'] = 'user'
            dict['content'] = input
            data_new['messages'].append(dict)
            res.append(InferRequest(**data_new))

        infer_requests = res
        # 批量推理
        infer_batch(engine, infer_requests)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in result:
                # print(item)
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')
            logging.info(f"infer success, file saved path: {output_path}")

    except Exception as e:
        logging.error(f"操作失败: {e}")