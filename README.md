import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lendo o arquivo em um DataFrame
df = pd.read_csv(r'C:\Users\micha\Downloads\ecommerce_estatistica.csv.csv')

# Visualização inicial
print(df.head())
print(df.info())
print(df.describe())

# Verificando valores nulos
print(df.isnull().sum())

# Correlação com quantidade vendida
print(df[['Qtd_Vendidos_Cod', 'N_Avaliações', 'Nota', 'Preço', 'Desconto']].corr())

sns.set(style='whitegrid')

# 1. Histograma - distribuição dos preços
plt.figure(figsize=(8, 5))
sns.histplot(df['Preço'], bins=20, kde=True)
plt.title('Histograma dos Preços')
plt.xlabel('Preço')
plt.ylabel('Frequência')
plt.show()

# 2. Gráfico de dispersão - avaliações x quantidade vendida
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='N_Avaliações', y='Qtd_Vendidos_Cod')
plt.title('Dispersão: Número de Avaliações x Quantidade Vendida')
plt.xlabel('Número de Avaliações')
plt.ylabel('Quantidade Vendida')
plt.show()

# 3. Mapa de calor - correlação entre variáveis numéricas
plt.figure(figsize=(10, 7))
colunas_corr = ['Nota', 'N_Avaliações', 'Desconto', 'Preço', 'Qtd_Vendidos_Cod',
                'Nota_MinMax', 'N_Avaliações_MinMax', 'Desconto_MinMax', 'Preço_MinMax']

corr = df[colunas_corr].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de Calor das Correlações')
plt.show()

# 4. Gráfico de barra - top 10 marcas mais frequentes
plt.figure(figsize=(10, 5))
top_marcas = df['Marca'].value_counts().head(10)

sns.barplot(x=top_marcas.values, y=top_marcas.index)
plt.title('Top 10 Marcas Mais Frequentes')
plt.xlabel('Quantidade de Produtos')
plt.ylabel('Marca')
plt.show()

# 5. Gráfico de pizza - distribuição dos materiais mais comuns
top_materiais = df['Material'].value_counts().head(6)

plt.figure(figsize=(7, 7))
plt.pie(top_materiais.values, labels=top_materiais.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribuição dos Principais Materiais')
plt.show()

# 6. Gráfico de densidade - distribuição dos descontos
plt.figure(figsize=(8, 5))
sns.kdeplot(df['Desconto'], fill=True)
plt.title('Gráfico de Densidade dos Descontos')
plt.xlabel('Desconto')
plt.ylabel('Densidade')
plt.show()

# 7. Gráfico de regressão - avaliações x quantidade vendida
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x='N_Avaliações', y='Qtd_Vendidos_Cod', scatter_kws={'alpha': 0.6})
plt.title('Regressão: Número de Avaliações x Quantidade Vendida')
plt.xlabel('Número de Avaliações')
plt.ylabel('Quantidade Vendida')
plt.show()
