import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Análise Exploratória
def perform_eda(df):
    print("\n=== Análise Exploratória de Dados ===")
    
    # Distribuição do Churn
    print("\nDistribuição de Churn:")
    print(df['churn'].value_counts(normalize=True))
    
    # Correlação
    print("\nCorrelações com Churn:")
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()['churn'].sort_values(ascending=False)
    print(correlation)


# 2. Pré-processamento
def preprocess_data(df):
    # Codificação
    le = LabelEncoder()
    categorical_cols = ['genero', 'localizacao', 'tipo_politica', 'nivel_cobertura']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Seleção de features
    features = ['idade', 'genero', 'localizacao', 'tempo_cliente_anos', 
                'tipo_politica', 'premio_mensal', 'nivel_cobertura',
                'sinistros_ultimo_ano', 'atraso_pagamento_dias',
                'satisfacao_cliente', 'numero_interacoes_atendimento']
    
    X = df[features]
    y = df['churn']
    
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Normalização
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, features

# 3. Modelagem Preditiva
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Avaliação
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n=== Desempenho do Modelo ===")
    print(classification_report(y_test, y_pred))
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nAUC-ROC: {roc_auc_score(y_test, y_proba):.2f}")
    
    return model

# 4. Identificação de Clientes de Risco
def identify_high_risk(model, df, features):

    X, _, _ = preprocess_data(df)
    
    # probabilidades de Churn
    probabilities = model.predict_proba(X)[:, 1]
    df['probabilidade_churn'] = probabilities
    
    # Top 10 clientes com maior risco
    high_risk = df.sort_values('probabilidade_churn', ascending=False).head(10)
    
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\n=== Fatores Mais Importantes para Churn ===")
    print(importance_df)
    
    # Explicação
    print("\n=== Clientes com Maior Risco de Churn ===")
    for idx, row in high_risk.iterrows():
        print(f"\nCliente ID: {row['customer_id']}")
        print(f"Probabilidade de Churn: {row['probabilidade_churn']:.2%}")
        print("Principais Razões:")
        
        # Comparar com valores médios
        if row['premio_mensal'] > df['premio_mensal'].median():
            print(f"- Prêmio mensal acima da média ({row['premio_mensal']:.2f} vs {df['premio_mensal'].median():.2f})")
        if row['satisfacao_cliente'] < 3:
            print(f"- Baixa satisfação ({row['satisfacao_cliente']} vs média {df['satisfacao_cliente'].mean():.1f})")
        if row['atraso_pagamento_dias'] > 0:
            print(f"- Atraso no pagamento ({row['atraso_pagamento_dias']} dias)")
        if row['sinistros_ultimo_ano'] > 0:
            print(f"- Sinistros recentes ({row['sinistros_ultimo_ano']})")
        if row['numero_interacoes_atendimento'] > df['numero_interacoes_atendimento'].median():
            print(f"- Muitas interações com atendimento ({row['numero_interacoes_atendimento']})")


if __name__ == "__main__":

    df = pd.read_csv('dados_clientes.csv')
    
    perform_eda(df)
    
    X, y, features = preprocess_data(df)
    
    model = train_model(X, y)
    
    identify_high_risk(model, df, features)