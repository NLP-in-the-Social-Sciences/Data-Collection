import json

def load_json_data(file: str) -> dict:
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f) 
    return (data)


def write_json_data(file: str, data: dict):
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)