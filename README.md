# LLM Unlearning Framework

A comprehensive framework for machine learning model unlearning using LoRA (Low-Rank Adaptation) adapters, supporting both targeted and untargeted unlearning approaches.

## 📁 Project Structure

```
my-LLM-Unlearning/
├── accelerate_config.yaml      # Accelerate training configuration
├── adapter/                    # LoRA adapter storage
├── config/                     # Model configurations
│   └── model_config.yaml
├── data/                       # Training datasets
│   ├── forget/                 # Forget dataset (data to unlearn)
│   └── retain/                 # Retain dataset (data to preserve)
├── dataset/
│   ├── data_module.py          # Dataset processing module
│   ├── __init__.py
│   └── __pycache__/
├── merged_model/               # Merged model storage
├── results/                    # Validation inference results
├── scores/                     # Evaluation scores storage
├── trainer/
│   ├── __init__.py
│   ├── losses.py               # Custom loss functions
│   └── __pycache__/
├── utils/
│   ├── __init__.py
│   ├── __pycache__/
│   └── utils.py               # Utility functions
├── extract_data.py            # Dataset splitting utility
├── evaluate.py                # Evaluation and scoring script
├── forget_idk_ap.py          # Targeted unlearning training
├── forget_me_gd.py           # Untargeted unlearning training
├── merge.py                  # Adapter merging utility
├── predict_demo.py           # Validation set inference
└── requirements.txt          # Python dependencies
```

## 🛠️ Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd my-LLM-Unlearning
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Data Preparation

### Dataset Structure

Organize your data into two main categories:

- **Forget Dataset** (`data/forget/`): Contains data that the model should "unlearn" or forget
- **Retain Dataset** (`data/retain/`): Contains data that the model should continue to remember

### Data Splitting

Use the provided data splitting utility:

```bash
python extract_data.py
```

## 🚄 Training

### Multi-GPU Training (Recommended)

Launch distributed training with default parameters using Accelerate:

```bash
accelerate launch forget_me_gd.py --config_file accelerate_config.yaml
```

### Training Parameters

The framework supports extensive customization through command-line arguments:

#### Basic Training Configuration

```bash
accelerate launch forget_me_gd.py \
    --config_file ./accelerate_config.yaml \
    --num_epochs 5 \
    --bs 1 \
    --max_length 512 \
    --lr 3e-5
```

#### Loss Function Configuration

```bash
--loss_type ME+GD \
--forget_coeff 0.7 \
--regularization_coeff 1.0
```

#### Data Configuration

```bash
--forget_data_path ./data/forget/forget.jsonl \
--retain_data_path ./data/retain/retain.jsonl \
--mask false
```

#### LoRA Configuration

```bash
--lora_r 16 \
--lora_alpha 32 \
--lora_dropout 0.05
```

### Complete Training Example

```bash
accelerate launch forget_me_gd.py \
    --config_file ./accelerate_config.yaml \
    --num_epochs 5 \
    --bs 1 \
    --max_length 400 \
    --loss_type ME+GD \
    --forget_data_path ./data/forget/forget.jsonl \
    --retain_data_path ./data/retain/retain.jsonl \
    --mask false \
    --lr 3e-5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --forget_coeff 1.0 \
    --regularization_coeff 1.0
```

### Training Methods

#### Untargeted Unlearning (ME+GD)

For general unlearning with gradient descent regularization:

```bash
accelerate launch forget_me_gd.py --config_file accelerate_config.yaml [additional parameters]
```

### Single GPU Training

If using a single GPU, you can run the scripts directly:

```bash
python forget_me_gd.py [parameters]
```

## ⚙️ Configuration

### Training Parameters Reference

| Parameter            | Description                                   | Default/Recommended Value  |
| -------------------- | --------------------------------------------- | -------------------------- |
| `--config_file`      | Accelerate configuration file path            | `./accelerate_config.yaml` |
| `--num_epochs`       | Number of training epochs                     | Dynamic based on dataset   |
| `--bs`               | Training batch size (limited by GPU memory)   | `1`                        |
| `--max_length`       | Maximum input sequence length                 | `512`                      |
| `--loss_type`        | Custom loss function type                     | `ME+GD`                    |
| `--forget_data_path` | Path to forget dataset                        | `./data/forget`            |
| `--retain_data_path` | Path to retain dataset                        | `./data/retain`            |
| `--mask`             | Whether to mask input questions               | `false`                    |
| `--lr`               | Learning rate (adjust based on dataset scale) | `3e-5`                     |

### Accelerate Configuration

Edit `accelerate_config.yaml` to customize your training setup:

- Number of GPUs
- Mixed precision settings
- Gradient accumulation steps

### Model Configuration

Modify `config/model_config.yaml` to adjust:

- Base model family

## 🧪 Evaluation and Testing

### Run Inference on Validation Set

```bash
python predict_demo.py \
	--model_path path/to/model\
	--data_path path/to/valid/dataset\
	--output_path ./results/output.jsonl\
	--model_type qwen2_5\
	--tensor_parallel_size 4\
```

### Evaluate Model Performance

```bash
python evaluate.py \
	--model_name model/used/for/judge \
	--test_data path/to/valid/dataset\
	--user_out_path ./results/output.json\
	--we_out_path output produced by the original model\
	--out_path path/for/scores
```

Results will be saved to:

- `results/`: Inference outputs
- `scores/`: Evaluation metrics and scores

## 🔧 Model Management

### Merge LoRA Adapters

After training, merge the LoRA adapters with the base model:

```bash
python merge.py
```

The merged model will be saved in the `merged_model/` directory.

## 📈 Workflow

1. **Prepare Data**: Organize forget and retain datasets
2. **Split Data**: Run `extract_data.py` if needed
3. **Configure**: Adjust configuration files as needed
4. **Train**: Execute unlearning training using Accelerate
5. **Merge**: Combine LoRA adapters with base model
6. **Deploy**: Use the merged model for inference
7. **Evaluate**: Run evaluation scripts to assess performance

## 🎯 Key Components

- **Custom Loss Functions**: ME+GD loss combining forget loss and regularization loss
- **Parameter Optimization**: Fine-tuned hyperparameters for memory-constrained environments
- **LoRA Configuration**: Optimized rank, alpha, and dropout settings
- **Data Processing**: Handled by `dataset/data_module.py`
- **Utilities**: Helper functions in `utils/utils.py`
- **LoRA Adapters**: Stored in `adapter/` directory

## 📋 Requirements

See `requirements.txt` for a complete list of dependencies. 

## 🙏 Acknowledgments

This repository is based on the codebase of the [TOFU Benchmark](https://github.com/locuslab/tofu/) and [A Closer Look at Machine Unlearning for Large Language Models](https://arxiv.org/abs/2410.08109). Thanks for their impressive works!