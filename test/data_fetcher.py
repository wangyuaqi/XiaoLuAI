import json

import requests
import pandas as pd


def fetch(url_):
    data_json = requests.get(url_, timeout=10).json()
    print(data_json)
    cols = data_json['cols']
    rows = data_json['rows']
    print([col['label'] for col in cols])


def trans_json(jsonpath):
    data = ''
    with open(jsonpath, mode='rt', encoding='utf-8') as f:
        data = ''.join(f.readlines())
    data_json = json.loads(data)
    cols = data_json['cols']
    rows = data_json['rows']

    data_df = []
    for row in rows:
        df_row = []
        for row_c_v in row['c']:
            df_row.append(row_c_v['v'])
        data_df.append(df_row)

    df = pd.DataFrame(data_df)
    df.to_excel('data.xlsx', sheet_name='Sheet1')


if __name__ == '__main__':
    trans_json('./data.json')
    """
    fetch(
        'http://arabidopsis.gmi.oeaw.ac.at:5000/DisplayResultsGene/fetchTopCandidateGenesFromOneResultOneGeneList?type_id'
        '=56&phenotype_method_id=11&analysis_method_id=2&list_type_id=0&max_rank=50')
    """
