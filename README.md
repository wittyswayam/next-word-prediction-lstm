# Next Word Prediction using LSTM (End-to-End NLP Project)

## Abstract
This project presents an end-to-end **Next Word Prediction** system built using a **Long Short-Term Memory (LSTM)** neural network.  
The model learns sequential word dependencies from a text corpus and predicts the most likely subsequent token given a partial sentence.  

The goal of this project is **not to achieve state-of-the-art language generation**, but to provide a **clear, interpretable, and deployable implementation of classical neural language modeling**, including data preprocessing, sequence modeling, inference, and web-based deployment.

---

## 1. Problem Definition
Given a sequence of words:

```

w₁, w₂, ..., wₙ

```

the task is to model the conditional probability:

```

P(wₙ₊₁ | w₁, w₂, ..., wₙ)

````

and predict the next most probable word based on learned linguistic patterns.

This formulation represents a **language modeling problem**, which forms the foundation of many NLP applications such as text generation, autocomplete, and speech recognition.

---

## 2. Motivation & Learning Objectives
This project was developed to achieve the following objectives:

- Understand **sequence modeling** in NLP
- Learn how **Recurrent Neural Networks (RNNs)** and **LSTMs** capture temporal dependencies
- Implement **tokenization, padding, and sequence generation**
- Train and serialize ML models for reuse
- Build an **interactive inference pipeline** using Streamlit
- Bridge the gap between **model training and real-world usage**

---

## 3. Dataset Description
The model is trained on a **quote-based text dataset** (`qoute_dataset.csv`).

### Dataset Characteristics
- Short, grammatically complete sentences
- Limited vocabulary size
- Clean, noise-free text

### Implications
While this dataset is suitable for:
- Demonstrating language modeling concepts
- Fast experimentation
- UI-driven inference

it is **not sufficient for learning complex linguistic structures** such as long-range dependencies or nuanced grammar.

This project therefore prioritizes **clarity and interpretability over linguistic richness**.

---

## 4. Data Preprocessing Pipeline
1. **Text Normalization**
   - Lowercasing
   - Removal of unnecessary whitespace

2. **Tokenization**
   - Each word is mapped to an integer index using `Tokenizer`
   - Vocabulary size is derived from corpus statistics

3. **Sequence Generation**
   - Input sequences are generated incrementally:
     ```
     [w₁] → w₂
     [w₁, w₂] → w₃
     ...
     ```

4. **Padding**
   - Sequences are padded to a fixed maximum length
   - Ensures consistent tensor dimensions for batch training

The tokenizer and maximum sequence length are serialized for inference consistency.

---

## 5. Model Architecture
The language model is implemented using a **stacked neural architecture**:

````

Embedding Layer
↓
LSTM Layer
↓
Fully Connected (Dense) Layer
↓
Softmax Output

```

### Architectural Rationale
- **Embedding Layer**: Learns dense vector representations of words
- **LSTM**: Captures sequential dependencies and mitigates vanishing gradients
- **Softmax Output**: Models probability distribution over the vocabulary

This architecture represents a **classical neural language model**, predating transformer-based approaches, and remains valuable for educational and conceptual understanding.

---

## 6. Training Strategy
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Training Objective**: Maximum likelihood estimation of the next word

Due to dataset size constraints, the model is optimized for **learning stability and convergence**, rather than aggressive generalization.

---

## 7. Inference Mechanism
During inference:
1. User input is tokenized using the trained tokenizer
2. Input sequence is padded to the learned maximum length
3. Model outputs a probability distribution over the vocabulary
4. The word with the highest probability (argmax) is selected as the prediction

This represents **greedy decoding**, chosen for simplicity and interpretability.

---

## 8. Web Application (Streamlit)
A lightweight **Streamlit web application** enables real-time interaction with the trained model.

### Features
- User-provided input sentence
- Instant next-word prediction
- Stateless and lightweight inference pipeline

The application demonstrates how trained ML models can be integrated into **user-facing systems**.

---

## 9. Project Structure
```

next-word-prediction-lstm/
│
├── app.py                     # Streamlit application for inference
├── lstm_model.h5               # Trained LSTM model
├── tokenizer.pkl               # Serialized tokenizer
├── max_len.pkl                 # Maximum sequence length
├── qoute_dataset.csv           # Training dataset
├── notebooks/                  # Training and experimentation notebooks
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

```

---

## 10. Limitations & Design Trade-offs
- Small dataset limits linguistic generalization
- Greedy decoding reduces output diversity
- LSTM-based architecture is computationally less expressive than transformers

These limitations are **acknowledged design choices**, aligned with the project’s educational goals.

---

## 11. Potential Extensions
- Training on large-scale corpora (Wikipedia, news data)
- Top-k or temperature-based sampling
- Bidirectional LSTM or attention mechanisms
- Transformer-based language models (BERT/GPT)
---

## 12. Key Learning Outcomes
- Practical understanding of neural language modeling
- Experience with sequential data pipelines
- Model serialization and reuse
- ML deployment using Streamlit
- Awareness of architectural trade-offs in NLP systems

---

## 13. Intended Audience
This project is intended for:
- Students learning NLP fundamentals
- ML beginners transitioning from software engineering
- Practitioners seeking an interpretable language modeling example

---
