import pickle
import sys

def model_test(attributes):
    # model = pickle.load(open('./saved_model/final_dataset_RandomForestFULLDATASET.sav', 'rb'))
    model = pickle.load(open('./saved_model/final_dataset_RandomForestFULLDATASET.sav', 'rb'))
    for attribute_set in attributes:
        result = model.predict([attribute_set])
        print(result)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <caminho_para_o_ficheiro>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        with open(file_path, 'r') as file:
            attributes_list = []
            for line in file:
                # Supondo que os atributos estão separados por vírgula
                attributes = list(map(float, line.strip().split(' ')))
                attributes_list.append(attributes)
            
        model_test(attributes_list)
    
    except FileNotFoundError:
        print(f"Ficheiro {file_path} não encontrado.")
        sys.exit(1)
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        sys.exit(1)
