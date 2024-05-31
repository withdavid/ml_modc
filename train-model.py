import pandas as pd
import numpy as np
import logging, os, time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import seaborn as sns

if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Configure the logger
log_folder = 'logs'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_filename = os.path.join(log_folder, f'training_session-{int(time.time())}.log')

# Set up logging to file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(message)s'))
logger.addHandler(file_handler)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

dtypes = {
    'Src IP': 'category',
    'Src Port': 'uint16',
    'Dst IP': 'category',
    'Dst Port': 'uint16',
    'Protocol': 'category',
    'Flow Duration': 'uint32',
    'Tot Fwd Pkts': 'uint32',
    'Tot Bwd Pkts': 'uint32',
    'TotLen Fwd Pkts': 'float32',
    'TotLen Bwd Pkts': 'float32',
    'Fwd Pkt Len Max': 'float32',
    'Fwd Pkt Len Min': 'float32',
    'Fwd Pkt Len Mean': 'float32',
    'Fwd Pkt Len Std': 'float32',
    'Bwd Pkt Len Max': 'float32',
    'Bwd Pkt Len Min': 'float32',
    'Bwd Pkt Len Mean': 'float32',
    'Bwd Pkt Len Std': 'float32',
    'Flow Byts/s': 'float32',
    'Flow Pkts/s': 'float32',
    'Flow IAT Mean': 'float32',
    'Flow IAT Std': 'float32',
    'Flow IAT Max': 'float32',
    'Flow IAT Min': 'float32',
    'Fwd IAT Tot': 'float32',
    'Fwd IAT Mean': 'float32',
    'Fwd IAT Std': 'float32',
    'Fwd IAT Max': 'float32',
    'Fwd IAT Min': 'float32',
    'Bwd IAT Tot': 'float32',
    'Bwd IAT Mean': 'float32',
    'Bwd IAT Std': 'float32',
    'Bwd IAT Max': 'float32',
    'Bwd IAT Min': 'float32',
    'Fwd PSH Flags': 'category',
    'Bwd PSH Flags': 'category',
    'Fwd URG Flags': 'category',
    'Bwd URG Flags': 'category',
    'Fwd Header Len': 'uint32',
    'Bwd Header Len': 'uint32',
    'Fwd Pkts/s': 'float32',
    'Bwd Pkts/s': 'float32',
    'Pkt Len Min': 'float32',
    'Pkt Len Max': 'float32',
    'Pkt Len Mean': 'float32',
    'Pkt Len Std': 'float32',
    'Pkt Len Var': 'float32',
    'FIN Flag Cnt': 'category',
    'SYN Flag Cnt': 'category',
    'RST Flag Cnt': 'category',
    'PSH Flag Cnt': 'category',
    'ACK Flag Cnt': 'category',
    'URG Flag Cnt': 'category',
    'CWE Flag Count': 'category',
    'ECE Flag Cnt': 'category',
    'Down/Up Ratio': 'float32',
    'Pkt Size Avg': 'float32',
    'Fwd Seg Size Avg': 'float32',
    'Bwd Seg Size Avg': 'float32',
    'Fwd Byts/b Avg': 'uint32',
    'Fwd Pkts/b Avg': 'uint32',
    'Fwd Blk Rate Avg': 'uint32',
    'Bwd Byts/b Avg': 'uint32',
    'Bwd Pkts/b Avg': 'uint32',
    'Bwd Blk Rate Avg': 'uint32',
    'Subflow Fwd Pkts': 'uint32',
    'Subflow Fwd Byts': 'uint32',
    'Subflow Bwd Pkts': 'uint32',
    'Subflow Bwd Byts': 'uint32',
    'Init Fwd Win Byts': 'uint32',
    'Init Bwd Win Byts': 'uint32',
    'Fwd Act Data Pkts': 'uint32',
    'Fwd Seg Size Min': 'uint32',
    'Active Mean': 'float32',
    'Active Std': 'float32',
    'Active Max': 'float32',
    'Active Min': 'float32',
    'Idle Mean': 'float32',
    'Idle Std': 'float32',
    'Idle Max': 'float32',
    'Idle Min': 'float32',
    'Label': 'category'
}

logger.info("Reading dataset..")

df = pd.read_csv('./dataset/final_dataset.csv',
     dtype=dtypes,
     parse_dates=['Timestamp'],
     usecols=[*dtypes.keys(), 'Timestamp'],
     engine='c',
     low_memory=True
     )

logger.info("Dataset was successfully read.")

def plot_confusion_matrix(matrix, class_names):

    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('./confusion_matrix_v3.png')
    logger.info("Confusion matrix image saved as 'confusion_matrix.png'.")

def treatDataset(df):
    logger.info("Started the process of treating data of the dataset...")
    # Drop columns that have only 1 value viewed
    colsToDrop = np.array(['Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'])

    # Drop columns where missing values are more than 40% and Drop rows where a column missing values are no more than 5%
    missing = df.isna().sum()
    missing = pd.DataFrame({'count': missing, '% of total': missing/len(df)*100}, index=df.columns)
    colsToDrop = np.union1d(colsToDrop, missing[missing['% of total'] >= 40].index.values)
    dropnaCols = missing[(missing['% of total'] > 0) & (missing['% of total'] <= 5)].index.values

    # Handling faulty data
    df['Flow Byts/s'].replace(np.inf, np.nan, inplace=True)
    df['Flow Pkts/s'].replace(np.inf, np.nan, inplace=True)
    dropnaCols = np.union1d(dropnaCols, ['Flow Byts/s', 'Flow Pkts/s'])

    # Drop the columns
    df.drop(columns=colsToDrop, inplace=True)
    df.dropna(subset=dropnaCols, inplace=True)
    logger.info("Process concluded with success. \n")
    return df

def train_and_evaluate_models(df):
    logger.info("Model Training:")
    features = ["Fwd Seg Size Avg", "Flow IAT Min", "Flow Duration", "Tot Fwd Pkts", "Pkt Size Avg", "Src Port", "Init Bwd Win Byts"]
    target = "Label"

    x = df.loc[:,features]
    y = df.loc[:,target]

    logger.info("Class distribution in the dataset: %s", y.value_counts())

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3)

    models = {
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=10000),
        "GaussianNB": GaussianNB(),
        "MLP": MLPClassifier(max_iter=1000)
    }

    best_model = None
    best_f1_score = 0
    best_model_name = ""

    for name, model in models.items():
        logger.info(f"Training model: {name}")
        if name == "MLP":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)


        score = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, pos_label='Benign') * 100
        recall = recall_score(y_test, y_pred, pos_label='Benign') * 100
        f1 = f1_score(y_test, y_pred, pos_label='Benign')*100

        logger.info("Accuracy of the model %s is: %f", name, score)
        logger.info("Precision of the model %s is: %f", name, precision)
        logger.info("Recall of the model %s is: %f", name, recall)
        logger.info(f"F1 score of the model {name} is: {f1}")

        if f1 > best_f1_score:
            best_f1_score = f1
            best_model = model
            best_model_name = name

    logger.info(f"\nBest model is {best_model_name} with F1 score: {best_f1_score}")

    if best_model is not None:
        y_pred = best_model.predict(x_test)

        matrix = confusion_matrix(y_test, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(matrix)

        plot_confusion_matrix(matrix, class_names=['Benign', 'Attack'])

        pickle.dump(best_model, open(f"./saved_model/bestmodel_v3.sav", 'wb'))

if __name__ == "__main__":
    dfTreated = treatDataset(df)
    train_and_evaluate_models(dfTreated)
