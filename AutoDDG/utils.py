# Adapted from:
# Zhang, H., Liu, Y., Hung, W.-L., Santos, A., & Freire, J. (2025).
# "AutoDDG: Automated Dataset Description Generation using Large Language Models".
# arXiv:2502.01050. https://doi.org/10.48550/arXiv.2502.01050

import pandas as pd

def get_sample(data_pd, sample_size, random_state=9):
    if sample_size <= len(data_pd):
        data_sample = data_pd.sample(sample_size, random_state=random_state)
    else:
        data_sample = data_pd
    sample_csv = data_sample.to_csv(index=False)
    sample_df = data_sample
    return sample_df, sample_csv

def flatten(data, parent_key='', sep='.'):
    """Aplana un diccionario o lista de forma recursiva."""
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten(v, new_key, sep=sep).items())
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(flatten(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, data))
    return dict(items)

def json_to_dataframe(json_input):
    """Convierte cualquier JSON (dict o list) en un DataFrame plano."""
    if isinstance(json_input, dict):
        json_input = [json_input]  # lo convertimos a lista para unificar
    flat_data = [flatten(record) for record in json_input]
    return pd.DataFrame(flat_data)
