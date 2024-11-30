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

1. Instale o Python 3.8+ e crie um ambiente virtual:
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/MacOS
   env\Scripts\activate     # Windows

pip install -r requirements.txt

# 📂 Estrutura do Repositório

desafio-tecnico-rag/
├── inputs/                  # Dados brutos e processados
├── notebooks/             # Análises exploratórias e experimentos
├── src/                   # Código-fonte principal
│   ├── evaluate_rag.py    # Scripts para treinamento e avaliação
│   ├── run_chatbot.py     # Aplicação interativa de teste
│   ├── generate_report.py # Geração do relatório técnico
├── requirements.txt       # Dependências do projeto
└── README.md              # Documentação do repositório

# 🧪 Métricas de Avaliação
Para comparar os métodos de RAG, as seguintes métricas foram utilizadas:

Precisão: Relevância dos chunks recuperados.
Tempo de Resposta: Latência entre consulta e resposta.
Escalabilidade: Desempenho com grandes volumes de dados.
Satisfação: Avaliação qualitativa dos resultados gerados.
Essas métricas foram escolhidas com base na relevância para tarefas de recuperação e geração.

# 📊 Resultados
Os resultados comparativos entre os métodos serão apresentados no relatório técnico disponível na pasta reports/.

Os principais pontos incluem:

Eficiência computacional.
Relevância dos resultados para diferentes tipos de consultas.
Justificativa da melhor solução com base nas métricas analisadas.

# 🤝 Como Contribuir
Sugestões e melhorias são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.

