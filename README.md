# assignment3_4995_UNI_yb2630

This repository contains a lightweight, decoder-only Transformer built entirely from scratch using PyTorch. The model is trained on the Tiny Shakespeare dataset to perform next-token prediction, demonstrating the core mechanics of Large Language Models (LLMs) within a heavily constrained environment.

## 🧠 Architecture & Features
This project implements a custom Transformer architecture adhering to strict assignment constraints (Vocab size ≤ 500) while integrating advanced optimization techniques to maximize performance.

* **Custom BPE Tokenization:** Trained a subword Byte Pair Encoding (BPE) tokenizer capped at 500 tokens to balance character-level granularity and word-level coherence.
* **Rotary Position Embeddings (RoPE):** Replaced standard sinusoidal positional encodings with RoPE to better capture relative token distances within the attention mechanism.
* **Weight Tying:** Shared weights between the token embedding layer and the output language modeling head to act as a powerful regularizer and reduce parameter count.
* **Cosine Annealing Scheduler:** Implemented a dynamic learning rate scheduler to stabilize late-stage training and prevent validation loss spikes.
* **RMSNorm:** Used Root Mean Square Normalization for improved training stability compared to standard LayerNorm.

## 📊 Hyperparameters
| Parameter | Value |
| :--- | :--- |
| **Vocab Size** | 500 |
| **Context Window (Block Size)** | 128 |
| **Embedding Dimension** | 128 |
| **Attention Heads** | 4 |
| **Transformer Layers** | 2 |
| **Dropout** | 0.20 |
| **Epochs** | 30 |

## 📈 Results
The model successfully avoided the "capacity cliff" (severe overfitting observed in larger models on this tiny dataset) by balancing regularization and model size. 

* **Final Training Loss:** ~3.03
* **Final Validation Loss:** ~3.89
* **Final Validation Perplexity (PPL):** ~48.9

Despite the highly restricted vocabulary, the model successfully learned structural text elements, including Shakespearean speaker tags (e.g., `C O M IN IUS :`) and coherent subword combinations.

## 📂 Repository Contents
* `transformer.py` / `transformer.ipynb`: The complete source code for the tokenizer, dataset preparation, model architecture, and training loop.
* `tinyshakespeare.txt`: The training corpus.

## 🛠️ Usage
Ensure you have the required dependencies installed:
`pip install torch tokenizers matplotlib seaborn`

Run the script to download the dataset, train the BPE tokenizer, initialize the model, and begin the 30-epoch training loop. Attention heatmap visualizations and sample text generations will output upon training completion.

Colab Link: https://colab.research.google.com/drive/1UtK4DGeF6pklrQdhxtunxjzPj5ixiCm8?usp=sharing
