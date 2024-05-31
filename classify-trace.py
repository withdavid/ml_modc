import pickle
import sys
import warnings

# Ignora os UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

def model_test(attributes, line_numbers):
    model = pickle.load(open('./saved_model/bestmodel_v3.sav', 'rb'))
    for attribute_set, line_number in zip(attributes, line_numbers):
        result = model.predict([attribute_set])
        print(f"Line {line_number}: {result[0]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Use: python3 classify-trace.py traces_file")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        with open(file_path, 'r') as file:
            attributes_list = []
            line_numbers = []
            for line_number, line in enumerate(file, start=1):
                # Atributos estão separados por espaços
                attributes = list(map(float, line.strip().split(' ')))
                attributes_list.append(attributes)
                line_numbers.append(line_number)
            
        model_test(attributes_list, line_numbers)
    
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error has occurred: {e}")
        sys.exit(1)
