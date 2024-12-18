
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import pymysql
import boto3
import os


def prediction():
    # define as chaves de acesso ao AWS
    os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAST6S64SLCYZ7BWXN'
    os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-2'

    # Define o endpoint
    client = boto3.client('rds', region_name='us-east-2')
    endpoint = 'mlet-b3.c3qk06ewapu6.us-east-2.rds.amazonaws.com'

    # Conecta com o banco de dados AWS RDS
    connection = pymysql.connect(
        host=endpoint,
        port=3306,
        user='admin',
        password='Fazcw5fo1!',
        db='mlet_b3'
    )

    # Carregar os dados
    data = pd.read_sql('SELECT * FROM TradeInformationConsolidated', connection)

    # fecha conexão com banco de dados
    connection.close()

    # Remover linhas com valores ausentes
    data.dropna(inplace=True)

    # Filtrar dados para a ação PETR4
    petr4_data = data[data['TckrSymb'] == 'PETR4']

    # Engenharia de Recursos
    petr4_data['RptDt'] = pd.to_datetime(petr4_data['RptDt'], infer_datetime_format=True)
    petr4_data.set_index('RptDt', inplace=True)

    # Análise Exploratória de Dados (EDA)
    plt.figure(figsize=(10, 6))
    sns.histplot(petr4_data['LastPric'], kde=True)
    plt.title('Distribuição dos Preços de Fechamento para PETR4')
    plt.xlabel('Preço de Fechamento')
    plt.ylabel('Frequência')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='RptDt', y='LastPric', data=petr4_data)
    plt.title('Preços de Fechamento ao Longo do Tempo para PETR4')
    plt.xlabel('Data')
    plt.ylabel('Preço de Fechamento')
    plt.xticks(rotation=45)
    plt.show()

    # Criar recursos de defasagem
    for lag in range(1, 6):
        petr4_data[f'Lag_{lag}'] = petr4_data['LastPric'].shift(lag)

    # Remover linhas com valores NaN criados pelos recursos de defasagem
    petr4_data.dropna(inplace=True)

    # Definir recursos e variável alvo
    X = petr4_data[[f'Lag_{lag}' for lag in range(1, 6)]]
    y = petr4_data['LastPric']

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar modelos com GridSearchCV para otimização de hiperparâmetros
    lr_model = LinearRegression()
    ridge_model = Ridge()
    lasso_model = Lasso()
    dt_model = DecisionTreeRegressor()
    rf_model = RandomForestRegressor()
    svm_model = SVR()

    # Definir parâmetros para GridSearchCV
    ridge_params = {'alpha': [0.1, 1.0, 10.0]}
    lasso_params = {'alpha': [0.1, 1.0, 10.0]}
    dt_params = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
    svm_params = {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}

    # Aplicar GridSearchCV para encontrar os melhores hiperparâmetros
    ridge_grid = GridSearchCV(ridge_model, ridge_params, cv=5)
    lasso_grid = GridSearchCV(lasso_model, lasso_params, cv=5)
    dt_grid = GridSearchCV(dt_model, dt_params, cv=5)
    rf_grid = GridSearchCV(rf_model, rf_params, cv=5)
    svm_grid = GridSearchCV(svm_model, svm_params, cv=5)

    # Treinar modelos
    lr_model.fit(X_train, y_train)
    ridge_grid.fit(X_train, y_train)
    lasso_grid.fit(X_train, y_train)
    dt_grid.fit(X_train, y_train)
    rf_grid.fit(X_train, y_train)
    svm_grid.fit(X_train, y_train)

    # Fazer previsões
    lr_pred = lr_model.predict(X_test)
    ridge_pred = ridge_grid.predict(X_test)
    lasso_pred = lasso_grid.predict(X_test)
    dt_pred = dt_grid.predict(X_test)
    rf_pred = rf_grid.predict(X_test)
    svm_pred = svm_grid.predict(X_test)

    # Avaliar modelos
    def evaluate_model(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, r2

    lr_mse, lr_r2 = evaluate_model(y_test, lr_pred)
    ridge_mse, ridge_r2 = evaluate_model(y_test, ridge_pred)
    lasso_mse, lasso_r2 = evaluate_model(y_test, lasso_pred)
    dt_mse, dt_r2 = evaluate_model(y_test, dt_pred)
    rf_mse, rf_r2 = evaluate_model(y_test, rf_pred)
    svm_mse, svm_r2 = evaluate_model(y_test, svm_pred)

    print(f"Linear Regression - MSE: {lr_mse}, R2: {lr_r2}")
    print(f"Ridge Regression - MSE: {ridge_mse}, R2: {ridge_r2}")
    print(f"Lasso Regression - MSE: {lasso_mse}, R2: {lasso_r2}")
    print(f"Decision Tree Regression - MSE: {dt_mse}, R2: {dt_r2}")
    print(f"Random Forest Regression - MSE: {rf_mse}, R2: {rf_r2}")
    print(f"SVM Regression - MSE: {svm_mse}, R2: {svm_r2}")

    # Selecionar o melhor modelo com base no R2
    best_model_name = 'Linear Regression'
    best_model_score = lr_r2

    if ridge_r2 > best_model_score:
        best_model_name = 'Ridge Regression'
        best_model_score = ridge_r2

    if lasso_r2 > best_model_score:
        best_model_name = 'Lasso Regression'
        best_model_score = lasso_r2

    if dt_r2 > best_model_score:
        best_model_name = 'Decision Tree Regression'
        best_model_score = dt_r2

    if rf_r2 > best_model_score:
        best_model_name = 'Random Forest Regression'
        best_model_score = rf_r2

    if svm_r2 > best_model_score:
        best_model_name = 'SVM Regression'
        best_model_score = svm_r2

    print(f"O melhor modelo é {best_model_name} com um R2 de {best_model_score}")

    # Aplicar o melhor modelo na base completa para prever os próximos 5 dias
    if best_model_name == 'Linear Regression':
        best_model = lr_model
    elif best_model_name == 'Ridge Regression':
        best_model = ridge_grid.best_estimator_
    elif best_model_name == 'Lasso Regression':
        best_model = lasso_grid.best_estimator_
    elif best_model_name == 'Decision Tree Regression':
        best_model = dt_grid.best_estimator_
    elif best_model_name == 'Random Forest Regression':
        best_model = rf_grid.best_estimator_
    else:
        best_model = svm_grid.best_estimator_

    # Prever os próximos 5 dias (exemplo simples usando o último valor conhecido como base)
    last_known_values = petr4_data.iloc[-1][[f'Lag_{lag}' for lag in range(1, 6)]].values.reshape(1, -1)
    predictions = []

    for _ in range(5):
        next_prediction = best_model.predict(last_known_values)[0]
        predictions.append(next_prediction)
        last_known_values = np.roll(last_known_values, -1)
        last_known_values[0][-1] = next_prediction

    print("Previsões para os próximos 5 dias:", predictions)

    # Preparar saída para um servidor r3 da Amazon (exemplo simples salvando em CSV)
    output_df = pd.DataFrame({
        'Date': pd.date_range(start=petr4_data.index.sort_values()[-1], periods=6, inclusive='right'),
        'Predicted_Close': predictions
    })

    output_df.to_sql(con=connection, name='TradeInformationPredict', if_exists='replace')

return output_df
