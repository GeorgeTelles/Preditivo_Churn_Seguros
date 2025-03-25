import pandas as pd
import numpy as np
import random
from random import choices

np.random.seed(42)
random.seed(42)

n_clientes = 30000

dados = {
    'customer_id': [],
    'idade': [],
    'genero': [],
    'localizacao': [],
    'tempo_cliente_anos': [],
    'tipo_politica': [],
    'premio_mensal': [],
    'nivel_cobertura': [],
    'sinistros_ultimo_ano': [],
    'atraso_pagamento_dias': [],
    'satisfacao_cliente': [],
    'numero_interacoes_atendimento': [],
    'churn': []
}

# Preenchimento dos dados
for id in range(1, n_clientes + 1):
    # Dados básicos
    dados['customer_id'].append(id)
    dados['idade'].append(int(np.random.normal(45, 10)))
    dados['genero'].append(choices(['M', 'F'], weights=[0.55, 0.45])[0])
    dados['localizacao'].append(choices(['Urbano', 'Suburbano', 'Rural'], weights=[0.6, 0.3, 0.1])[0])
    
    # Tempo como cliente
    dados['tempo_cliente_anos'].append(round(np.random.exponential(scale=5)) + 1)
    
    # Tipo de política
    tipo_politica = choices(['Compreensiva', 'Terceiros', 'Basica'], weights=[0.4, 0.4, 0.2])[0]
    dados['tipo_politica'].append(tipo_politica)
    
    # Prêmio mensal baseado em idade e tipo de política
    base_premio = 100 + (dados['idade'][-1] * 0.5)
    if tipo_politica == 'Compreensiva':
        base_premio *= 2.5
    elif tipo_politica == 'Terceiros':
        base_premio *= 1.8
    dados['premio_mensal'].append(round(base_premio + np.random.normal(0, 20), 2))
    
    # Nível de cobertura
    dados['nivel_cobertura'].append(choices(['Alta', 'Media', 'Baixa'], 
                                          weights=[0.3, 0.5, 0.2])[0])
    
    # Sinistros no último ano
    dados['sinistros_ultimo_ano'].append(np.random.poisson(0.5))
    
    # Atrasos de pagamento
    if np.random.rand() < 0.7: 
        dados['atraso_pagamento_dias'].append(0)
    else:
        dados['atraso_pagamento_dias'].append(np.random.poisson(15))
    
    # Satisfação do cliente (1-5)
    dados['satisfacao_cliente'].append(choices([1, 2, 3, 4, 5], 
                                             weights=[0.1, 0.2, 0.3, 0.25, 0.15])[0])
    
    dados['numero_interacoes_atendimento'].append(np.random.poisson(2))
    
    # Cálculo de probabilidade de Churn
    churn_score = (
        -0.1 * dados['idade'][-1] +
        0.3 * (dados['premio_mensal'][-1]/100) +
        0.4 * dados['atraso_pagamento_dias'][-1] +
        -0.5 * dados['satisfacao_cliente'][-1] +
        -0.2 * dados['tempo_cliente_anos'][-1] +
        0.3 * dados['sinistros_ultimo_ano'][-1] +
        0.2 * dados['numero_interacoes_atendimento'][-1] +
        np.random.normal(0, 0.5)
    )
    
    # Conversão para probabilidade usando função logística
    prob_churn = 1 / (1 + np.exp(-churn_score))
    dados['churn'].append(1 if prob_churn > 0.5 else 0)

df = pd.DataFrame(dados)

# valores extremos
df['idade'] = df['idade'].clip(18, 80)
df['premio_mensal'] = df['premio_mensal'].clip(100, 600)
df['atraso_pagamento_dias'] = df['atraso_pagamento_dias'].clip(0, 60)

# Salvar em CSV
df.to_csv('dados_clientes.csv', index=False)

print("Base de dados históricos criada com sucesso!")
print(f"Tamanho do dataset: {df.shape}")
print(f"Taxa de Churn: {df['churn'].mean():.2%}")