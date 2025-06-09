import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# Diretórios para salvar artefatos
OUTPUT_IMAGE_DIR = 'static/images'
OUTPUT_MODEL_DIR = 'models'
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

print("Iniciando o projeto de Machine Learning aplicado ao Varejo - Árvore de Decisão...")

# 1. Análise Exploratória de Dados
df = pd.read_csv('data/dataset.csv', sep=';')
df['Data_Pedido'] = pd.to_datetime(df['Data_Pedido'], format='%d/%m/%Y', errors='coerce')
df['Ano'] = df['Data_Pedido'].dt.year
df['Mes'] = df['Data_Pedido'].dt.month
df['DiaSemana'] = df['Data_Pedido'].dt.dayofweek
df['Trimestre'] = df['Data_Pedido'].dt.quarter
df.dropna(subset=['Ano', 'Mes', 'DiaSemana', 'Trimestre'], inplace=True)
df[['Ano', 'Mes', 'DiaSemana', 'Trimestre']] = df[['Ano', 'Mes', 'DiaSemana', 'Trimestre']].astype(int)

# Visualizações
plt.figure(figsize=(30, 15))
sns.countplot(data=df, y='Categoria', order=df['Categoria'].value_counts().index, palette='viridis')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMAGE_DIR, 'distribuicao_categoria.png'))
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(x=df['Segmento'].value_counts().index, y=df['Segmento'].value_counts().values, palette='plasma')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMAGE_DIR, 'vendas_por_segmento.png'))
plt.close()

vendas_por_mes = df.groupby(['Ano', 'Mes'])['Valor_Venda'].sum().reset_index()
vendas_por_mes['Data'] = pd.to_datetime(vendas_por_mes['Ano'].astype(str) + '-' + vendas_por_mes['Mes'].astype(str))
plt.figure(figsize=(30, 15))
sns.lineplot(x='Data', y='Valor_Venda', data=vendas_por_mes, marker='o')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMAGE_DIR, 'vendas_ao_longo_do_tempo.png'))
plt.close()

# 2. Preparação dos Dados
numerical_features = ['Valor_Venda', 'Ano', 'Mes', 'DiaSemana', 'Trimestre']
categorical_features = ['Segmento', 'Pais', 'Estado', 'Cidade', 'SubCategoria']
target_feature = 'Categoria'
X = df[numerical_features + categorical_features]
y = df[target_feature]

le_categoria = LabelEncoder()
y_encoded = le_categoria.fit_transform(y)
class_names = le_categoria.classes_

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 3. Treinamento do Modelo
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=10, random_state=42))
])
model_pipeline.fit(X_train, y_train)
y_pred_dt = model_pipeline.predict(X_test)

# 4. Avaliação
a_accuracy = accuracy_score(y_test, y_pred_dt)
print("Acurácia:", a_accuracy)

report_dt = classification_report(y_test, y_pred_dt, target_names=class_names)
print(report_dt)

cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(32, 16))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMAGE_DIR, 'confusion_matrix.png'))
plt.close()

# 5. Interpretabilidade
dt_classifier_trained = model_pipeline.named_steps['classifier']
plt.figure(figsize=(40, 20))
plot_tree(dt_classifier_trained, filled=True, max_depth=3, class_names=class_names)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMAGE_DIR, 'arvore_decisao.png'))
plt.close()

# 6. Salvamento dos Artefatos
pickle.dump(model_pipeline, open(os.path.join(OUTPUT_MODEL_DIR, 'modelo_arvore_decisao.pkl'), 'wb'))
pickle.dump(le_categoria, open(os.path.join(OUTPUT_MODEL_DIR, 'label_encoder_categoria.pkl'), 'wb'))

print("Artefatos salvos com sucesso nas pastas 'models' e 'static/images'.")
