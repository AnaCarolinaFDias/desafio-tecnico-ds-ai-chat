# desafio-tecnico-ds-ai-chat
Descrição Este repositório contém a implementação de um desafio técnico voltado para ciência de dados (DS) com foco em chatbots baseados em inteligência artificial (IA).

# **Desafio Técnico - Implementação e Avaliação de Métodos de RAG**

Este repositório contém a solução do Desafio Técnico para implementar, avaliar e comparar dois métodos distintos de Recuperação Aumentada por Geração (RAG). O objetivo é identificar a abordagem mais eficaz com base em métricas apropriadas, justificando as escolhas metodológicas e os resultados obtidos.

---

## 📋 **Sobre o Desafio**

Recuperação Aumentada por Geração (RAG) é uma técnica que combina a recuperação de informações relevantes de uma base de conhecimento com geração de respostas usando modelos de linguagem natural. Este desafio exige: 

1. Implementação de dois métodos de RAG:  
   - Um método **baseline** (simples recuperação dos *n* chunks mais similares ao contexto).  
   - Outro método de RAG de escolha livre, com justificativa técnica.

2. Avaliação e comparação entre os dois métodos, considerando métricas bem definidas e fundamentadas.  

3. Elaboração de um relatório técnico consolidado, apresentando:  
   - Metodologia.  
   - Métricas aplicadas.  
   - Resultados comparativos.  
   - Análise e justificativa da solução mais eficaz.

---

## 🚀 **Objetivo**

Identificar e justificar a solução de RAG mais eficiente e eficaz, com base em métricas técnicas relevantes. 

---

## 🛠️ **Como Utilizar Este Repositório**

### **Pré-requisitos**

1. Instale o Python e crie um ambiente virtual:
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/MacOS
   env\Scripts\activate     # Windows

pip install -r requirements.txt

# 📂 Estrutura do Repositório

desafio-tecnico-ds-ai-chat/
├── inputs/                         # Dados brutos e processados
├── langchain_collection/           # Vector Store criado a partir do Chroma
├── results/                        # Resultados e contextos gerados 
├── functions.py/                   # Módulo python com as funções criadas 
├── GeneratingRags.py/              # Módulo python para preparação e criação dos métodos RAG e geração de resultados
├── EvaluatingRags.py/              # Módulo python para análise dos resultados
├── evaluate_rag_notebook.ipynb     # Notebook contendo todo o processo de análise usado para construição e validação
├── logs                            # Logs par debug das implementações e resultados
├── requirements.txt                # Dependências do projeto
└── README.md                       # Documentação do repositório
