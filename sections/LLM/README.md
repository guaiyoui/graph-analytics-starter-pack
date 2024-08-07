  - [3: AIGC-LLM](#3-aigc-llm)
    - [3.1 Survey of AIGC-LLM](#31-survey-of-aigc-llm)
    - [3.2 Theory of AIGC-LLM](#32-theory-of-aigc-llm)
    - [3.3 Prompt Learning](#33-prompt-learning)
      - [Prompt **Engineering Techniques**](#prompt-engineering-techniques)
      - [In-context Learning](#in-context-learning)
      - [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
      - [Reasoning with Large Language Models](#reasoning-with-large-language-models)
      - [Multimodal Prompt](#multimodal-prompt)
      - [Evaluation \& Reliability](#evaluation--reliability)
      - [Others](#others)
    - [3.4 Foundation Models](#34-foundation-models)
      - [3.4.1 Encoder-only Architecture](#341-encoder-only-architecture)
      - [3.4.2 Decoder-only Architecture](#342-decoder-only-architecture)
      - [3.4.3 Encoder-decoder Architecture](#343-encoder-decoder-architecture)
      - [3.4.4 Other](#344-other)
    - [3.5 Related Repos](#35-related-repos)
    - [3.6 Datasets of LLM-AIGC](#36-datasets-of-llm-aigc)
    - [3.7 Tools for LLM-AIGC](#37-tools-for-llm-aigc)
      - [Open-Source LLMs](#open-source-llms)
      - [Prompt Learning](#prompt-learning)
      - [CoT](#cot)
      - [Development](#development)
      - [ChatBots](#chatbots)

<p id="AIGCLLM"></p>

## 3: AIGC-LLM

### 3.1 Survey of AIGC-LLM

- [**Augmented Language Models: a Survey](https://doi.org/10.48550/arXiv.2302.07842), Arxiv, 2023.02.15**
- [**A Survey for In-context Learning](https://doi.org/10.48550/arXiv.2301.00234), Arxiv, 2022.12.31**
- [**Reasoning with Language Model Prompting: A Survey](https://doi.org/10.48550/arXiv.2212.09597), Arxiv, 2022.12.19**
- [**Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://doi.org/10.1145/3560815), Arxiv, 2021.07.28**
- [**Emergent Abilities of Large Language Models](https://doi.org/10.48550/arXiv.2206.07682), Arxiv, 2022.06.15**
- [**Towards Reasoning in Large Language Models: A Survey](https://doi.org/10.48550/arXiv.2212.10403), Arxiv, 2022.12.20**

### 3.2 Theory of AIGC-LLM

- **[A Mathematical Exploration of Why Language Models Help Solve Downstream Tasks](https://arxiv.org/abs/2010.03648), 2020.10.7**
- **[Why Do Pretrained Language Models Help in Downstream Tasks? An Analysis of Head and Prompt Tuning](https://arxiv.org/abs/2106.09226), 2021.6.17**

### 3.3 Prompt Learning

#### Prompt **Engineering Techniques**

- **[Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data](https://doi.org/10.48550/arXiv.2302.12822)** （**2023.02.24**）
- **[Guiding Large Language Models via Directional Stimulus Prompting](https://doi.org/10.48550/arXiv.2302.11520)** （**2023.02.22**）
- **[Progressive Prompts: Continual Learning for Language Models](https://doi.org/10.48550/arXiv.2301.12314), 2023.01.29**
- **[Batch Prompting: Efficient Inference with Large Language Model APIs](https://doi.org/10.48550/arXiv.2301.08721)** （**2023.01.19**）
- **[One Embedder, Any Task: Instruction-Finetuned Text Embeddings](https://doi.org/10.48550/arXiv.2212.09741)** （**2022.12.19**）
- **[Successive Prompting for Decomposing Complex Questions](https://doi.org/10.48550/arXiv.2212.04092)** （**2022.12.08**）
- **[Promptagator: Few-shot Dense Retrieval From 8 Examples](https://doi.org/10.48550/arXiv.2209.11755)** （**2022.09.23**）
- **[Black-box Prompt Learning for Pre-trained Language Models](https://arxiv.org/abs/2201.08531)** （**2022.01.21**）
- **[Design Guidelines for Prompt Engineering Text-to-Image Generative Models](https://doi.org/10.1145/3491102.3501825)** （**2021.09.14**）
- **[Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm](https://doi.org/10.1145/3411763.3451760)** （**2021.02.15**）
- [**Making Pre-trained Language Models Better Few-shot Learners](https://doi.org/10.18653/v1/2021.acl-long.295), ACL, 2021.01.01**
- [**Eliciting Knowledge from Language Models Using Automatically Generated Prompts](https://doi.org/10.18653/v1/2020.emnlp-main.346), EMNLP, 2020.10.29**
- **[Automatically Identifying Words That Can Serve as Labels for Few-Shot Text Classification](https://doi.org/10.5282/UBM/EPUB.74034)** （**2020.10.26**）

#### In-context Learning

- **[Larger language models do in-context learning differently](https://doi.org/10.48550/arXiv.2303.03846)** （**2023.03.07**）
- **[Language Model Crossover: Variation through Few-Shot Prompting](https://doi.org/10.48550/arXiv.2302.12170)** （**2023.02.23**）
- **[How Does In-Context Learning Help Prompt Tuning?](https://doi.org/10.48550/arXiv.2302.11521)** （**2023.02.22**）
- **[Large Language Models Are Implicitly Topic Models: Explaining and Finding Good Demonstrations for In-Context Learning](https://doi.org/10.48550/arXiv.2301.11916)** （**2023.01.27**）
- **[Transformers as Algorithms: Generalization and Stability in In-context Learning](https://arxiv.org/abs/2301.07067)** （**2023.01.17**）
- **[OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization](https://doi.org/10.48550/arXiv.2212.12017), Meta,**（**2022.12.22**）
- [**Finetuned Language Models are Zero-Shot Learners](https://arxiv.org/abs/2109.01652),** ICLR, （2021.9.3）
    
    FLAN，多任务 instruction tuning
    
- [**Learning To Retrieve Prompts for In-Context Learning](https://arxiv.org/abs/2112.08633),** NAACL, （2022.12.16）
    
    Prompt 选择对模型效果有影响，之前的方法利用相似度的方式挑选合适的样本，本文提出用一个单独的模型对训练集中的每个样例进行评分，选取最合适的作为 demonstration
    

#### Parameter-Efficient Fine-Tuning (PEFT)

- [**LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS**](https://arxiv.org/pdf/2106.09685.pdf)
    
    **LoRA** 
    
- [**Prefix-Tuning: Optimizing Continuous Prompts for Generation**](https://aclanthology.org/2021.acl-long.353/)
    
    **Prefix Tuning** 
    
- [**GPT Understands, Too**](https://arxiv.org/pdf/2103.10385.pdf)
    
    **P-Tuning V1** 
    
- [**P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**](https://arxiv.org/pdf/2110.07602.pdf)
    
    **P-Tuning V2** 
    
- [**The Power of Scale for Parameter-Efficient Prompt Tuning**](https://arxiv.org/pdf/2104.08691.pdf)
    
    **Prompt Tuning** 
    

#### Reasoning with Large Language Models

- [**Automatic Chain of Thought Prompting in Large Language Models**](https://arxiv.org/abs/2210.03493)
    
    Auto-CoT
    
- **[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)**
    
    Manual-CoT
    
- **[Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916), NeurIPS, 2022**
    
    Zero-CoT
    

#### Multimodal Prompt

- [**Multimodal Chain-of-Thought Reasoning in Language Models**](https://arxiv.org/pdf/2302.00923.pdf)

#### Evaluation & Reliability

#### Others

- RPT: Relational Pre-trained Transformer Is Almost All You Need towards Democratizing Data Preparation, VLDB, 2021
- Can Foundation Models Wrangle Your Data?, VLDB, 2023

### 3.4 Foundation Models


#### 3.4.1 Encoder-only Architecture

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (2018.10.11)
- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) （2019.09.26）
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) （2019.07.26）
- [ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2112.12731)  (2021.12.23)

#### 3.4.2 Decoder-only Architecture

- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) （2023.03.15）
- GPT-3 [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) （2020.05.28）
- [JURASSIC-1: TECHNICAL DETAILS AND EVALUATION](https://assets.website-files.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf) (2021.08)
- Gopher [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446) （2021.12.08）
- [LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) （2022.01.20）
- Chinchilla [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556) （2022.03.29）
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311.pdf) （2022.04.05）
- [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) (2022.11.09)
- [OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization](https://arxiv.org/abs/2212.12017) (2022.12.22)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) （2023.02.27）

#### 3.4.3 Encoder-decoder Architecture

- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) （2019.10.29）
- T5 [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) （2019.10.23）
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) （2021.01.11）

#### 3.4.4 Other
- [GLM-130B: An Open Bilingual Pre-trained Model](https://arxiv.org/abs/2210.02414)  (2022.10.05)


### 3.5 Related Repos

- [Awesome-LLM: a curated list of Large Language Model](https://github.com/Hannibal046/Awesome-LLM)  
- [Awesome resources for in-context learning and prompt engineering: Mastery of the LLMs such as ChatGPT, GPT-3, and FlanT5, with up-to-date and cutting-edge updates](https://github.com/EgoAlpha/prompt-in-context-learning) 
- [This repository contains a hand-curated resources for Prompt Engineering with a focus on Generative Pre-trained Transformer (GPT), ChatGPT, PaLM etc](https://github.com/promptslab/Awesome-Prompt-Engineering) 
- [A trend starts from "Chain of Thought Prompting Elicits Reasoning in Large Language Models"](https://github.com/Timothyxxx/Chain-of-ThoughtsPapers) 
- [Collection of papers and resources on Reasoning in Large Language Models, including Chain-of-Thought, Instruction-Tuning, and others.](https://github.com/atfortes/LLM-Reasoning-Papers) 


### 3.6 Datasets of LLM-AIGC

- [Alpaca dataset from Stanford, cleaned and curated](https://github.com/gururise/AlpacaDataCleaned)  
- [Alpaca is a dataset of 52,000 instructions and demonstrations generated by OpenAI's `text-davinci-003` engine. This instruction data can be used to conduct instruction-tuning for language models and make the language model follow instruction better.](https://huggingface.co/datasets/tatsu-lab/alpaca/tree/main/data) 


### 3.7 Tools for LLM-AIGC

- [Awesome-LLM: a curated list of Large Language Model](https://github.com/Hannibal046/Awesome-LLM)  
- [Awesome resources for in-context learning and prompt engineering: Mastery of the LLMs such as ChatGPT, GPT-3, and FlanT5, with up-to-date and cutting-edge updates](https://github.com/EgoAlpha/prompt-in-context-learning) 
- [This repository contains a hand-curated resources for Prompt Engineering with a focus on Generative Pre-trained Transformer (GPT), ChatGPT, PaLM etc](https://github.com/promptslab/Awesome-Prompt-Engineering) 
- [A trend starts from "Chain of Thought Prompting Elicits Reasoning in Large Language Models"](https://github.com/Timothyxxx/Chain-of-ThoughtsPapers) 
- [Collection of papers and resources on Reasoning in Large Language Models, including Chain-of-Thought, Instruction-Tuning, and others.](https://github.com/atfortes/LLM-Reasoning-Papers) 


#### Open-Source LLMs

- [Inference code for LLaMA models](https://github.com/facebookresearch/llama)
- [Code and documentation to train Stanford's Alpaca models, and generate the data.](https://github.com/tatsu-lab/stanford_alpaca)
- [Port of Facebook's LLaMA model in C/C++](https://github.com/ggerganov/llama.cpp)
- [Locally run an Instruction-Tuned Chat-Style LLM](https://github.com/antimatter15/alpaca.cpp)
- [Instruct-tune LLaMA on consumer hardware](https://github.com/tloen/alpaca-lora)
- [骆驼:A Chinese finetuned instruction LLaMA](https://github.com/LC1332/Chinese-alpaca-lora)
- [Alpaca-LoRA as Chatbot service](https://github.com/deep-diver/Alpaca-LoRA-Serve)
- his fine-tunes the [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) model on the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset using a Databricks notebook [The repo](https://github.com/databrickslabs/dolly)
- [ChatGLM-6B：开源双语对话语言模型 | An Open Bilingual Dialogue Language Model](https://github.com/THUDM/ChatGLM-6B)
- GPT-J 6B is a transformer model trained using Ben Wang's **[Mesh Transformer JAX](https://github.com/kingoflolz/mesh-transformer-jax/)**. "GPT-J" refers to the class of model, while "6B" represents the number of trainable parameters. [EleutherAI/gpt-j-6B · Hugging Face](https://huggingface.co/EleutherAI/gpt-j-6B)
    
- [一种平价的chatgpt实现方案, 基于ChatGLM-6B + LoRA](https://github.com/mymusise/ChatGLM-Tuning)
- [Open Academic Research on Improving LLaMA to SOTA LLM](https://github.com/AetherCortex/Llama-X)

#### Prompt Learning

- [An Open-Source Framework for Prompt-Learning](https://github.com/thunlp/OpenPrompt)
- [PEFT: State-of-the-art Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)

#### CoT

- [Benchmarking LLM reasoning performance w. chain-of-thought prompting](https://github.com/FranxYao/chain-of-thought-hub)

#### Development

- [Examples and guides for using the OpenAI API](https://github.com/openai/openai-cookbook)
- [A gradio web UI for running Large Language Models like GPT-J 6B, OPT, GALACTICA, LLaMA, and Pygmalion.](https://github.com/oobabooga/text-generation-webui)
- [GUI for ChatGPT API](https://github.com/GaiZhenbiao/ChuanhuChatGPT)
- [LlamaIndex (GPT Index) is a project that provides a central interface to connect your LLM's with external data.](https://github.com/jerryjliu/llama_index)
- [The ChatGPT Retrieval Plugin lets you easily search and find personal or work documents by asking questions in everyday language.](https://github.com/openai/chatgpt-retrieval-plugin)
- [Building applications with LLMs through composability](https://github.com/hwchase17/langchain)
- [Containers for machine learning](https://github.com/replicate/cog)
    
    

#### ChatBots

- [An open-source ChatGPT UI.](https://github.com/mckaywrigley/chatbot-ui) 
- [Create UIs for your machine learning model in Python in 3 minutes](https://github.com/gradio-app/gradio)
- [A web interface for chatting with Alpaca through llama.cpp. Fully dockerized, with an easy to use API.](https://github.com/nsarrazin/serge)
- [ChatLLaMA 📢 Open source implementation for LLaMA-based ChatGPT runnable in a single GPU. 15x faster training process than ChatGPT](https://github.com/juncongmoo/chatllama)
- [Locally running, hands-free ChatGPT](https://github.com/yakGPT/yakGPT)
- [An editor made for programming with AI](https://github.com/getcursor/cursor)
- [ChatGPT 学术优化](https://github.com/binary-husky/chatgpt_academic)
- [myGPTReader is a slack bot that can read any webpage, ebook, video(YouTube) or document and summarize it with chatGPT. It can also talk to you via voice using the content in the channel.](https://github.com/madawei2699/myGPTReader)
- [Use ChatGPT to summarize the arXiv papers.](https://github.com/kaixindelele/ChatPaper)
- [基于 ChatGPT API 的划词翻译浏览器插件和跨平台桌面端应用 - Browser extension and cross-platform desktop application for translation based on ChatGPT API.](https://github.com/yetone/openai-translator)
- [LLM Chain for answering questions from documents with citations](https://github.com/whitead/paper-qa)
- [Grounded search engine (i.e. with source reference) based on LLM / ChatGPT / OpenAI API. It supports web search, file content search etc.](https://github.com/michaelthwan/searchGPT)
- [An open-source LLM based research assistant that allows you to have a conversation with a research paper](https://github.com/mukulpatnaik/researchgpt)
- [A simple command-line interface tool that allows you to interact with ChatGPT from OpenAI.](https://github.com/p208p2002/heygpt)

<p id="GraphSimilarityComputation"></p>