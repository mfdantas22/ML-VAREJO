import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

print("Iniciando a criação do novo dataset sintético (em português)...")

# --- 1. Definir os dados base para as colunas ---
num_rows = 5000 # Número de linhas no dataset

# Dados Categóricos em português
segmentos = ['Consumidor', 'Corporativo', 'Home Office']
paises = ['Brasil'] # Mantendo apenas Brasil para simplificar
estados_cidades = {
    'SP': ['São Paulo', 'Campinas', 'Ribeirão Preto', 'Santos'],
    'RJ': ['Rio de Janeiro', 'Niterói', 'Duque de Caxias'],
    'MG': ['Belo Horizonte', 'Uberlândia', 'Juiz de Fora'],
    'RS': ['Porto Alegre', 'Caxias do Sul'],
    'PR': ['Curitiba', 'Londrina'],
    'BA': ['Salvador', 'Feira de Santana'],
    'DF': ['Brasília'],
    'CE': ['Fortaleza'],
    'PE': ['Recife']
}

# Categorias e Subcategorias em português
categorias_subcategorias = {
    'Móveis': ['Cadeiras', 'Mesas', 'Estantes', 'Mobiliário Diversos'],
    'Materiais de Escritório': ['Papel', 'Fichários', 'Armazenamento', 'Arte', 'Envelopes', 'Fixadores', 'Suprimentos', 'Etiquetas', 'Aparelhos'],
    'Tecnologia': ['Telefones', 'Máquinas', 'Acessórios', 'Copiadoras']
}

# Gerar IDs únicos
product_ids = [f'PROD-{i:05d}' for i in range(500)] # 500 produtos únicos
customer_ids = [f'CLI-{i:04d}' for i in range(1000)] # 1000 clientes únicos

# Gerar datas de pedido (últimos 3 anos)
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)
date_range = [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(num_rows)]

# --- 2. Gerar o DataFrame ---
data = {
    'ID_Pedido': [f'BR-{random.randint(2023, 2025)}-{i:06d}' for i in range(num_rows)],
    'Data_Pedido': [d.strftime('%d/%m/%Y') for d in date_range],
    'ID_Cliente': random.choices(customer_ids, k=num_rows),
    'Segmento': random.choices(segmentos, k=num_rows),
    'Pais': random.choices(paises, k=num_rows)
}

df = pd.DataFrame(data)

# Gerar Cidade e Estado de forma consistente
estados = list(estados_cidades.keys())
df['Estado'] = random.choices(estados, k=num_rows)
df['Cidade'] = df['Estado'].apply(lambda e: random.choice(estados_cidades[e]))

# Gerar Categoria, SubCategoria e ID_Produto de forma consistente
categorias_list = list(categorias_subcategorias.keys())
df['Categoria'] = random.choices(categorias_list, weights=[0.3, 0.5, 0.2], k=num_rows) # Proporção para categorias

# Usar a categoria para escolher uma subcategoria válida
subcategorias_geradas = []
id_produtos_gerados = []
for index, row in df.iterrows():
    cat = row['Categoria']
    subcat = random.choice(categorias_subcategorias[cat])
    subcategorias_geradas.append(subcat)
    id_produtos_gerados.append(random.choice(product_ids)) # Reutiliza IDs de produto

df['SubCategoria'] = subcategorias_geradas
df['ID_Produto'] = id_produtos_gerados

# Gerar Valor_Venda (com alguma variação por categoria)
def generate_sales_value(category):
    if category == 'Móveis':
        return max(10.0, round(np.random.normal(300.00, 150.00), 2))
    elif category == 'Materiais de Escritório':
        return max(1.0, round(np.random.normal(50.00, 30.00), 2))
    elif category == 'Tecnologia':
        return max(50.0, round(np.random.normal(800.00, 400.00), 2))
    return max(1.0, round(np.random.normal(150.00, 100.00), 2))

df['Valor_Venda'] = df['Categoria'].apply(generate_sales_value)

# Reordenar as colunas para corresponder à descrição
df = df[[
    'ID_Pedido', 'Data_Pedido', 'ID_Cliente', 'Segmento', 'Pais',
    'Cidade', 'Estado', 'ID_Produto', 'Categoria', 'SubCategoria',
    'Valor_Venda'
]]

# --- 3. Salvar o Dataset ---
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'dataset.csv')

df.to_csv(output_path, index=False)

print(f"\nDataset sintético com {num_rows} linhas criado e salvo em '{output_path}'.")
print("\nPrimeiras 5 linhas do novo dataset:")
print(df.head())
print("\nInformações do novo dataset:")
df.info()

print("\nCriação do dataset concluída!")