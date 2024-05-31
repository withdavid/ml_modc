import pandas as pd

# Colunas necessárias para o modelo
columns_needed = ["Fwd Seg Size Avg", "Flow IAT Min", "Flow Duration", "Tot Fwd Pkts", "Pkt Size Avg", "Src Port", "Init Bwd Win Byts"]

# Ler os dados de entrada
input_data = pd.read_csv("raw_traces.csv")
input_data_filtered = input_data[columns_needed]

# Guarda os dados filtrados em um novo arquivo
input_data_filtered.to_csv("traces.txt", index=False, header=False, sep=' ')
