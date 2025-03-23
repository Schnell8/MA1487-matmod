import csv
import numpy as np
import scipy.stats as stats
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

def parse_smhi_data(data_path, delimiter=';'):
    parsed_file_name = data_path[:-4] + '-parsed.csv'
    with open(data_path, 'r') as in_file, open(parsed_file_name, 'w') as out_file:
        reader = csv.reader(in_file, delimiter=delimiter)
        write = csv.writer(out_file, delimiter = delimiter)
        
        parsed_header = False
        for i, row in enumerate(reader):
            if len(row) > 0 and row[0] == 'Datum':
                parsed_header = True
            if parsed_header:
                write.writerow(row)

def clear_smhi_data_pandas(df):
    df['Datum'] = pd.to_datetime(df['Datum'].astype(str) + ' ' + df['Tid (UTC)'])
    # Remove unused data columns
    df.drop(['Tid (UTC)', 'Unnamed: 4', 'Tidsutsnitt:', 'Kvalitet'], inplace=True, axis=1)
    df = df.set_index('Datum')
    return df

def filter_data(df):
    # Filter rows with date 12-24 and time 12:00:00
    df_filtered = df[df.index.month == 12]  # Filter by December
    df_filtered = df_filtered[df_filtered.index.day == 24]  # Filter by 24st day
    df_filtered = df_filtered[df_filtered.index.hour == 12]  # Filter by 12:00:00

    # Filter rows with years from 1990 to 2019
    df_filtered = df_filtered[(df_filtered.index.year >= 1990) & (df_filtered.index.year <= 2019)]
    
    return df_filtered

############################################################################################################
#TASK 1

#LULEÅ
data_path_lulea = r'lulea.csv'
parsed_path_lulea = r'lulea-parsed.csv'
# Parse smhi data. Removes header data
parse_smhi_data(data_path_lulea)
# Load data from parsed csv
df_lulea = pd.read_csv(parsed_path_lulea, delimiter=';', low_memory=False)
# Parse date time and clear irrelevant columns
df_lulea = clear_smhi_data_pandas(df_lulea)
df_lulea = df_lulea.rename(columns={'Lufttemperatur' : 'Temp-Luleå'})
filtered_df_lulea = filter_data(df_lulea)

#MALMÖ
data_path_malmo = r'malmo.csv'
parsed_path_malmo = r'malmo-parsed.csv'
# Parse smhi data. Removes header data
parse_smhi_data(data_path_malmo)
# Load data from parsed csv
df_malmo = pd.read_csv(parsed_path_malmo, delimiter=';', low_memory=False)
# Parse date time and clear irrelevant columns
df_malmo = clear_smhi_data_pandas(df_malmo)
df_malmo = df_malmo.rename(columns={'Lufttemperatur' : 'Temp-Malmö'})
filtered_df_malmo = filter_data(df_malmo)

#STOCKHOLM
data_path_stockholm = r'stockholm.csv'
parsed_path_stockholm = r'stockholm-parsed.csv'
# Parse smhi data. Removes header data
parse_smhi_data(data_path_stockholm)
# Load data from parsed csv
df_stockholm = pd.read_csv(parsed_path_stockholm, delimiter=';', low_memory=False)
# Parse date time and clear irrelevant columns
df_stockholm = clear_smhi_data_pandas(df_stockholm)
df_stockholm = df_stockholm.rename(columns={'Lufttemperatur' : 'Temp-Stockholm'})
filtered_df_stockholm = filter_data(df_stockholm)

# Concatinate to one dataframe and rename columns
df = pd.concat([filtered_df_lulea, filtered_df_malmo, filtered_df_stockholm], axis=1/1) # enbart använda 1 gör resterande kod mörkare, mycket märkligt..

# Drop NaN rows to have equal data from all sites
df = df.dropna()

def task_one():
    print(df.head(n=7))
    # Plotta data
    plt.figure(figsize=(10, 6))
    plt.plot(df.index.year, df['Temp-Luleå'], 'o-', label='Luleå')
    plt.plot(df.index.year, df['Temp-Malmö'], 'o-', label='Malmö')
    plt.plot(df.index.year, df['Temp-Stockholm'], 'o-', label='Stockholm')
    plt.title('Temperatur Luleå, Malmö, Stockholm')
    plt.xlabel('Datum')
    plt.ylabel('Temperatur (°C)')
    plt.legend()

############################################################################################################
#TASK 2

# Medeltemperatur
mean_temperatures = df.mean(axis=0)
mean_temperatures_rounded = mean_temperatures.round(2)

# standardavvikelse temperatur
std_dev_temperatures = df.std(axis=0)
std_dev_temperatures_rounded = std_dev_temperatures.round(2)

# Max temperatur
max_temperatures = df.max(axis=0)

# Min temperatur
min_temperatures = df.min(axis=0)

# Korrelation mellan städer
correlation = df.corr()
correlation_rounded = correlation.round(2)

def task_two():
    print("\nMedelvärde:")
    print(mean_temperatures_rounded.to_string())
    print("\nStandardavvikelse:")
    print(std_dev_temperatures_rounded.to_string())
    print("\nMax temperaturer:")
    print(max_temperatures.to_string())
    print("\nMin temperaturer:")
    print(min_temperatures.to_string())
    print("\nKorrelation mellan städerna:")
    print(correlation_rounded)

    # Skapa heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_rounded, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Korrelation mellan temperaturer för städerna')

############################################################################################################
#TASK 3

def task_three():
    # Skapa ett lådagram för temperaturdata
    plt.figure(figsize=(10, 6))
    df.boxplot(column=['Temp-Luleå', 'Temp-Malmö', 'Temp-Stockholm'])
    plt.title('Lådagram för temperatur i Luleå, Malmö, Stockholm')
    plt.ylabel('Temperatur (°C)')

############################################################################################################
#TASK 4

x = df.index.year.values.reshape(-1, 1)
y = df['Temp-Malmö'].values

# Skapa en dataframe
df_temp_malmo = pd.DataFrame({'Year': x.flatten(), 'Temperature': y.flatten()})

# Skapa linjär regression
model = LinearRegression().fit(x, y)

# Koefficienter
a = model.intercept_ #skärning
b = model.coef_[0] #lutning

# Prediktera temperatur
y_pred = model.predict(x)

# MSE
mse = np.mean((y_pred - y)**2) # medelvärdet av kvadraten på skillnaden, mått på hur nära de predikterade värdena är de verkliga

def task_four():
    print(f"Koefficient a: {a}")
    print(f"Koefficient b: {b}")
    print(f"MSE: {mse}") # ju lägre ju mer tillförlitlig

    # Plotta linjär regression med konfidensintervall
    plt.figure(figsize=(10, 6))
    sns.lmplot(x='Year', y='Temperature', data=df_temp_malmo, ci=95, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'}, height=6, aspect=1.4)
    plt.xlabel('År')
    plt.ylabel('Temperatur (°C)')
    plt.title('Linjär regression för temperatur i Malmö med konfidensintervall')
    plt.legend(['Data', 'Linjär regression', 'Konfidensintervall'])

############################################################################################################
#TASK 5

# Logaritmisk transformation
log_x = np.log(x)

# Skapa linjär regression av logaritmerad data
log_model = LinearRegression().fit(log_x, y)

# Prediktera
log_pred = log_model.predict(log_x)

draw_exp_model = log_model.predict(log_x.reshape(-1, 1))

# Räkna ut MSE
log_mse = np.mean((log_pred - y)**2)

#####

# Exponentiell modell i y
log_y = np.log(np.abs(y))

# Skapa linjär regressiion av logaritmerad data
log_y_model = LinearRegression().fit(x, log_y)

# Prediktera
log_pred_y = log_y_model.predict(x)

y_log = log_y_model.predict(x.reshape(-1, 1))

# Beräkna residualer
log_residual_y = y - np.exp(log_pred_y)

# Skriv ut MSE
mse_exp_y = np.mean(log_residual_y**2)

def task_five():
    print(f"MSE för log transformerad y: {mse_exp_y}")
    print(f"MSE för log transformerad x: {log_mse}")
    print(f"MSE för linjär regression: {mse}") # lägst vilket gör den bättre lämpad

    # Visualisera samtliga modeller
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x, np.exp(y_log), color='green', linestyle='--', label='Exponentiell modell (y)')
    plt.plot(np.exp(log_x), draw_exp_model, color='cyan', label='Exponentiell modell (x)')
    plt.plot(x, y_pred, color='red', linestyle='-.', label='Linjär regression')
    plt.legend()
    plt.xlabel('År')
    plt.ylabel('Temperatur (°C)')
    plt.title('Prediktioner av temperatur (exponentiell modell i y)')

############################################################################################################
#TASK 6

def task_six():
    # Redidualvarians
    k = 2
    n = len(x)

    # Beräkna residualvarians (ju lägre ju bättre)
    residual_variance_linjär = np.sum((y - y_pred)**2) / (n - k)
    residual_variance_exp_y = np.sum((y - np.exp(log_pred_y))**2) / (n - k)
    residual_variance_exp_x = np.sum((y - log_pred)**2) / (n - k) # lägst vilket gör den bättre lämpad

    # Beräkna R^2 (ju närmare 1 ju bättre förklarar modellen variationen i datat)
    var_y = np.var(y)
    r2_linjär = 1 - residual_variance_linjär / var_y
    r2_exp_y = 1 - residual_variance_exp_y / var_y
    r2_exp_x = 1 - residual_variance_exp_x / var_y # närmast 1 vilket för den bättre lämpad

    print("Var(R)")
    print(f"Residualvarians för exponentiell modell i y: {residual_variance_exp_y}")
    print(f"Residualvarians för exponentiell modell i x: {residual_variance_exp_x}")
    print(f"Residualvarians för linjär regression: {residual_variance_linjär}\n")

    print("R^2")
    print(f"R^2 för exponentiell modell i y: {r2_exp_y}")
    print(f"R^2 för exponentiell modell i x: {r2_exp_x}")
    print(f"R^2 för linjär regression: {r2_linjär}")

    # QQ-plottar
    plt.figure(figsize=(18,6))

    plt.subplot(1,3,1)
    stats.probplot(y - y_pred, dist='norm', plot=plt)
    plt.title("Linjär regression")

    plt.subplot(1,3,2)
    stats.probplot(y - np.exp(log_pred_y), dist='norm', plot=plt)
    plt.title("Exponentiell modell i y")

    plt.subplot(1,3,3)
    stats.probplot(y - log_pred, dist='norm', plot=plt)
    plt.title("Exponentiell modell i x")