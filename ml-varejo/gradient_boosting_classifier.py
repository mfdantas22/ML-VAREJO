"""
Script para implementar e avaliar um modelo de Gradient Boosting Classifier
para predição da Categoria de Produto no dataset de vendas.

Este script executa as seguintes etapas:
1. Carregamento e Preparação dos Dados (similar ao main.py)
2. Treinamento do Modelo Gradient Boosting
3. Avaliação do Modelo (Acurácia, Relatório de Classificação, Matriz de Confusão)
4. Análise de Interpretabilidade (Importância das Features)
5. Salvamento de gráficos de resultados
6. Salvamento do Modelo e Pré-processadores para uso futuro

Autor: Seu Nome
Data: 07/06/2025 (data de criação/adaptação)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib # Importar joblib para salvar e carregar modelos

# Criar diretório para salvar imagens se não existir
os.makedirs('imagens', exist_ok=True)
# Criar diretório para salvar modelos se não existir
os.makedirs('models', exist_ok=True)


print("Iniciando o projeto de Machine Learning aplicado ao Varejo - Gradient Boosting...")

# --- 1. Carregamento e Preparação dos Dados ---
print("\n--- 1. Carregamento e Preparação dos Dados ---")
try:
    df = pd.read_csv('data/dataset.csv')
    print("Dataset 'data/dataset.csv' carregado com sucesso!")
except FileNotFoundError:
    print("Erro: 'data/dataset.csv' não encontrado. Certifique-se de que o arquivo está no mesmo diretório do script.")
    exit()

# Converter Data_Pedido para datetime e extrair características temporais
df['Data_Pedido'] = pd.to_datetime(df['Data_Pedido'], format='%d/%m/%Y')
df['Ano'] = df['Data_Pedido'].dt.year
df['Mes'] = df['Data_Pedido'].dt.month
df['DiaSemana'] = df['Data_Pedido'].dt.dayofweek
df['Trimestre'] = df['Data_Pedido'].dt.quarter
print("Coluna 'Data_Pedido' convertida para tipo datetime. Características temporais (Ano, Mes, DiaSemana, Trimestre) extraídas.")


# Codificar variáveis categóricas
print("\nCodificando variáveis categóricas...")
categorical_cols = ['Segmento', 'Pais', 'Cidade', 'Estado', 'Categoria', 'SubCategoria']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le # Guardar o encoder para possível uso futuro
    print(f" - Coluna '{col}' codificada para numérico.")

print("\nPrimeiras 5 linhas do dataset após codificação categórica:")
print(df.head())

print("\nTipos de dados após codificação categórica:")
print(df.info())

# Definição da variável alvo e features
# A variável alvo será 'Categoria', como na árvore de decisão
y = df['Categoria']
features = ['Segmento', 'Pais', 'Cidade', 'Estado', 'SubCategoria', 'Valor_Venda', 'Ano', 'Mes', 'DiaSemana', 'Trimestre']
X = df[features]
print(f"\nVariável Alvo (y): 'Categoria'")
print(f"Features (X) selecionadas: {features}")

# Padronização das features numéricas
print("\nPadronizando features numéricas (usando StandardScaler)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=features) # Converter de volta para DataFrame para manter os nomes das colunas
print("Features padronizadas com sucesso.")
print("\nPrimeiras 5 linhas das features (X) padronizadas:")
print(X.head())

# Divisão dos dados em treino e teste
print("\nDividindo os dados em conjuntos de treino e teste (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Tamanho do conjunto de treino (X_train): {X_train.shape}")
print(f"Tamanho do conjunto de teste (X_test): {X_test.shape}")
print(f"Tamanho do conjunto de treino (y_train): {y_train.shape}")
print(f"Tamanho do conjunto de teste (y_test): {y_test.shape}")

print("\n--- Fim da Etapa de Preparação dos Dados ---")

# --- 2. Implementação e Treinamento do Modelo (Gradient Boosting) ---
print("\n--- 2. Implementação e Treinamento do Modelo (Gradient Boosting) ---")
print("Modelo escolhido: GradientBoostingClassifier")
# Parâmetros comuns para Gradient Boosting. Experimente ajustá-los!
# learning_rate: contribuição de cada árvore
# n_estimators: número de árvores a construir
# max_depth: profundidade máxima de cada árvore base
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
print(f"Parâmetros iniciais: n_estimators={gb_classifier.n_estimators}, learning_rate={gb_classifier.learning_rate}, max_depth={gb_classifier.max_depth}, random_state=42")

print("\nTreinando o modelo de Gradient Boosting...")
gb_classifier.fit(X_train, y_train)
print("Modelo treinado com sucesso!")

print("\nRealizando previsões no conjunto de teste...")
y_pred_gb = gb_classifier.predict(X_test)
print("Previsões realizadas.")

print("\n--- Fim da Etapa de Implementação e Treinamento do Modelo ---")

# --- 3. Avaliação do Modelo (Gradient Boosting) ---
print("\n--- 3. Avaliação do Modelo (Gradient Boosting) ---")
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Acurácia: {accuracy_gb:.4f}")

print("\nRelatório de Classificação:")
# Para o classification_report, é útil ter os nomes das classes originais
# Vamos decodificar os rótulos de 'Categoria' para exibir no relatório
target_names = label_encoders['Categoria'].inverse_transform(sorted(y.unique()))
print(classification_report(y_test, y_pred_gb, target_names=target_names))

print("\nMatriz de Confusão:")
cm_gb = confusion_matrix(y_test, y_pred_gb)
print(cm_gb)

# Plotar e salvar a Matriz de Confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Matriz de Confusão - Gradient Boosting')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('./imagens/matriz_confusao_gradient_boosting.png')
plt.close() # Fechar o plot para liberar memória
print("Matriz de Confusão salva em 'imagens/matriz_confusao_gradient_boosting.png'.")

print("\n--- Fim da Etapa de Avaliação do Modelo ---")

# --- 4. Interpretabilidade do Modelo (Gradient Boosting) ---
print("\n--- 4. Interpretabilidade do Modelo (Gradient Boosting) ---")

# Importância das Features
print("\nImportância das Features:")
feature_importances_gb = pd.Series(gb_classifier.feature_importances_, index=features).sort_values(ascending=False)
print(feature_importances_gb)

# Plotar e salvar a Importância das Features
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_gb.values, y=feature_importances_gb.index, palette='viridis')
plt.title('Importância das Features - Gradient Boosting')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('./imagens/importancia_features_gradient_boosting.png')
plt.close() # Fechar o plot para liberar memória
print("Gráfico de Importância das Features salvo em 'imagens/importancia_features_gradient_boosting.png'.")

print("\n--- Fim da Etapa de Interpretabilidade do Modelo ---")

# --- 5. Salvamento do Modelo e Pré-processadores ---
print("\n--- 5. Salvamento do Modelo e Pré-processadores ---")
os.makedirs('models', exist_ok=True) # Garante que o diretório 'models' existe

# Salvar o modelo Gradient Boosting
joblib.dump(gb_classifier, './models/gradient_boosting_model.pkl')
print("Modelo de Gradient Boosting salvo em './models/gradient_boosting_model.pkl'.")

# Salvar o StandardScaler
joblib.dump(scaler, './models/scaler_gb.pkl')
print("StandardScaler salvo em './models/scaler_gb.pkl'.")

# Salvar TODOS os LabelEncoders para o Gradient Boosting (com um nome diferente)
joblib.dump(label_encoders, './models/all_label_encoders_gb.pkl') # Nome diferente!
print("Todos os LabelEncoders para GB salvos em './models/all_label_encoders_gb.pkl'.")

# Salvar o LabelEncoder específico da Categoria para decodificação
joblib.dump(label_encoders['Categoria'], './models/label_encoder_categoria_gb.pkl') # Nome diferente!
print("LabelEncoder da Categoria para GB salvo em './models/label_encoder_categoria_gb.pkl'.")


# Salvar a lista de features usadas para o treino
joblib.dump(features, './models/features_list_gb.pkl')
print("Lista de features para GB salva em './models/features_list_gb.pkl'.")

print("\n--- Fim da Etapa de Salvamento ---")
print("\n--- Fim do Projeto de Machine Learning aplicado ao Varejo - Gradient Boosting ---")
print("Você pode verificar os gráficos na pasta 'imagens/' e os modelos na pasta 'models/'.")