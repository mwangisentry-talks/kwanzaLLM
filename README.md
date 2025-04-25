# kwanzaLLM
 A journey towards LLMs
# 🧠 kwanzaLLM

**kwanzaLLM** is an experimental lightweight **Large Language Model (LLM)** project focused on understanding the internals of LLM architecture, training, and inference. Built from scratch for educational and research purposes.

## 🎯 Goals

- Demystify the inner workings of LLMs  
- Build a minimal transformer-based architecture  
- Train and evaluate on small-scale datasets  
- Experiment with tokenization, attention, and fine-tuning  

## ⚙️ Features

- Custom tokenizer implementation  
- Scaled dot-product attention mechanism  
- Transformer encoder/decoder modules  
- Simple training loop using PyTorch  
- Lightweight inference engine for text generation  

## 🧪 Experiments & Demos

- Tiny dataset training (e.g., Shakespeare, dialogues)
- Tokenization strategies (byte-pair encoding, character-level)
- Hyperparameter tuning for low-resource settings
- Future: Instruction fine-tuning & quantization

## 🛠️ Tech Stack

- Python 3  
- PyTorch  
- NumPy  
- Matplotlib (for training visualizations)

## 🚀 Getting Started

```bash
git clone https://github.com/mngugi/kwanzaLLM.git
cd kwanzaLLM
python train.py --config configs/small_transformer.yaml

📚 References
- Attention Is All You Need (Vaswani et al.)

- nanoGPT by Andrej Karpathy

- The Annotated Transformer

** 📜 License
- MIT License
