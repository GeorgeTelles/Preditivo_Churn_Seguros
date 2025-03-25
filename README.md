# Modelo Preditivo de Churn com Identificação de Clientes de Alto Risco em Seguros  

Este código implementa uma solução completa para análise e predição de churn (cancelamento) em clientes de seguros, utilizando machine learning. O fluxo inclui etapas de exploração de dados, pré-processamento, modelagem e geração de insights acionáveis.

---

## **Etapas do Projeto**  

### 1. **Análise Exploratória de Dados (EDA)**  
- Gera estatísticas sobre a distribuição de `churn` na base.  
- Calcula correlações entre variáveis numéricas e a taxa de cancelamento.  
- Identifica padrões iniciais para orientar a modelagem.  

### 2. **Pré-processamento**  
- **Codificação:** Transforma variáveis categóricas (`genero`, `localizacao`, etc.) em numéricas com `LabelEncoder`.  
- **Seleção de Features:** Define variáveis preditoras como `idade`, `premio_mensal`, `satisfacao_cliente`, entre outras.  
- **Tratamento de Dados:**  
  - Imputa valores ausentes usando a mediana (`SimpleImputer`).  
  - Normaliza os dados com `StandardScaler`.  

### 3. **Modelagem Preditiva**  
- **Algoritmo:** Random Forest com ajuste para classes desbalanceadas (`class_weight='balanced'`).  
- **Avaliação:**  
  - Métricas: Precisão, recall, F1-score, matriz de confusão e AUC-ROC.  
  - Estratégia: Divisão estratificada de dados (80% treino / 20% teste).  

### 4. **Identificação de Clientes de Risco**  
- **Classificação:** Lista os 10 clientes com maior probabilidade de churn.  
- **Explicabilidade:**  
  - Destaca fatores críticos (ex.: `premio_mensal` acima da média, `atraso_pagamento_dias`).  
  - Mostra a importância das variáveis no modelo (feature importance).  

---

## **Tecnologias e Métodos**  
- **Bibliotecas:**  
  - `Pandas` e `NumPy` para manipulação de dados.  
  - `Scikit-learn` para machine learning e pré-processamento.  
  - `Seaborn`/`Matplotlib` para visualização (não explicitado no código, mas sugerido pelas importações).  
- **Técnicas:**  
  - Tratamento de dados desbalanceados.  
  - Interpretabilidade de modelo (explicações baseadas em regras).  

---

## **Saída do Modelo**  
1. **Relatório de Desempenho:**  
   - Classificação detalhada (precision, recall, F1).  
   - AUC-ROC para avaliação da capacidade de discriminação.  

2. **Lista de Ações Prioritárias:**  
   - Clientes com risco crítico de churn, acompanhados de razões específicas para intervenção (ex.: baixa satisfação, histórico de sinistros).  

---

**Ideal para:** Empresas de seguros que buscam reduzir o cancelamento de clientes com estratégias baseadas em dados e machine learning.  
