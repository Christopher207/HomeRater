import json

def preprocess_txt_to_json(input_file, output_file):
    properties = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            #print(parts[6])
            if len(parts) < 9:
                continue
            prop = {
                'id': parts[0],
                'tipo': parts[1],
                'contrato': parts[2],
                'ubicacion': parts[3],
                'titulo': parts[4],
                'precio': parts[5],
                'coords': [float(n) for n in parts[6].replace("('","").replace("')","").replace("'","").split(", ")],
                'imagen': parts[7],
                'descripcion': parts[8]
            }
            properties.append(prop)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(properties, f, ensure_ascii=False)

if __name__ == '__main__':
    preprocess_txt_to_json('depaAlqUrbaniaConsolidado.txt', 'static/data/properties.json')
