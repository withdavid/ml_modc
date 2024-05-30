import os
import time
import pandas as pd
import numpy as np
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Configure the logger
log_folder = 'logs'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_filename = os.path.join(log_folder, f'training_phase-{int(time.time())}.log')

# Set up logging to file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(message)s'))
logger.addHandler(file_handler)

def reduce_mem_usage(df):
    """Iterate through all the columns of a dataframe and modify the data type to reduce memory usage."""
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            try:
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            except Exception as e:
                logger.error(f"Error converting column {col}: {e}")
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f'Memory usage after optimization is: {end_mem:.2f} MB')
    logger.info(f'Decreased by {(100 * (start_mem - end_mem) / start_mem):.1f}%')
    
    return df

def treatDataset(df):
    try:
        logger.info("Started the process of treating data of the dataset...")
        df = reduce_mem_usage(df)

        # Identificar colunas com mais de 40% de valores ausentes para remoção
        missing = df.isna().sum()
        missing = pd.DataFrame({'count': missing, '% of total': missing / len(df) * 100}, index=df.columns)
        colsToDrop = missing[missing['% of total'] >= 40].index.values
        dropnaCols = missing[(missing['% of total'] > 0) & (missing['% of total'] <= 5)].index.values

        # Tratamento de valores infinitos
        if 'Flow Byts/s' in df.columns:
            df['Flow Byts/s'] = df['Flow Byts/s'].replace(np.inf, np.nan)
            dropnaCols = np.union1d(dropnaCols, ['Flow Byts/s'])
        if 'Flow Pkts/s' in df.columns:
            df['Flow Pkts/s'] = df['Flow Pkts/s'].replace(np.inf, np.nan)
            dropnaCols = np.union1d(dropnaCols, ['Flow Pkts/s'])

        # Remover colunas identificadas para drop
        colsToDrop = [col for col in colsToDrop if col in df.columns]
        df.drop(columns=colsToDrop, inplace=True)
        df.dropna(subset=dropnaCols, inplace=True)
        logger.info("Process concluded with success.\n")
        return df
    except Exception as e:
        logger.error(f"Error in treating dataset: {e}")
        raise

def train_classifier(name, model, x_train, y_train, x_test, y_test, positive_label):
    try:
        start_time = time.time()
        model.fit(x_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=positive_label)
        recall = recall_score(y_test, y_pred, pos_label=positive_label)
        f1 = f1_score(y_test, y_pred, pos_label=positive_label)

        return {
            "name": name,
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "training_time": training_time
        }
    except Exception as e:
        logger.error(f"Error training classifier {name}: {e}")
        raise

def train_model(df, modelname_prefix, output_folder, csv_file):
    try:
        logger.info(f"Starting model training for file: {csv_file}")
        
        # Amostragem do dataset para acelerar o treino
        df = df.sample(frac=0.2, random_state=42)  # Usando 10% do dataset para treinamento

        features = [
            "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
            "Flow Byts/s", "Flow Pkts/s", "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean",
            "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean"
        ]
        target = "Label"

        df.columns = df.columns.str.strip()

        missing_features = [feature for feature in features if feature not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}. Skipping model training for this dataset.")
            return

        x = df.loc[:, features]
        y = df.loc[:, target]

        labels = y.unique()
        positive_label = labels[1] if len(labels) > 1 else labels[0]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # Seleção de features
        selector = SelectKBest(f_classif, k='all')
        x_train = selector.fit_transform(x_train, y_train)
        x_test = selector.transform(x_test)

        classifiers = {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=2000),
            "GaussianNB": GaussianNB(),
            "MLP": MLPClassifier(max_iter=2000)
        }

        best_model = None
        best_score = 0
        best_model_name = ""
        best_time = 0

        with ThreadPoolExecutor() as executor:
            futures = []
            for name, model in classifiers.items():
                futures.append(executor.submit(train_classifier, name, model, x_train, y_train, x_test, y_test, positive_label))

            for future in as_completed(futures):
                result = future.result()
                logger.info(f"Accuracy: {result['accuracy']}, Precision: {result['precision']}, Recall: {result['recall']}, F1 Score: {result['f1']}, Training time: {result['training_time']:.2f} seconds, Model: {result['name']}")

                if result['f1'] > best_score:
                    best_score = result['f1']
                    best_model = result['model']
                    best_model_name = result['name']
                    best_time = result['training_time']

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        filename = os.path.join(output_folder, f'{modelname_prefix}_{best_model_name}_v2.sav')
        pickle.dump(best_model, open(filename, 'wb'))
        logger.info(f"Best model: {best_model_name} with F1 Score: {best_score}, training time: {best_time:.2f} seconds, saved successfully as {filename}\n")
    except Exception as e:
        logger.error(f"Error in training model for file {csv_file}: {e}")
        raise

input_folder = 'dataset'
output_folder = './saved_model'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Train individual models
for csv_file in os.listdir(input_folder):
    if csv_file.endswith('.csv'):
        try:
            file_path = os.path.join(input_folder, csv_file)
            logger.info(f'Processing file: {file_path}')
            df = pd.read_csv(file_path, low_memory=False)
            df = treatDataset(df)
            modelname_prefix = os.path.splitext(csv_file)[0]
            train_model(df, modelname_prefix, output_folder, csv_file)
        except Exception as e:
            logger.error(f"Error processing file {csv_file}: {e}")

logger.info("All CSV files processed, models trained, and best models saved successfully.")
