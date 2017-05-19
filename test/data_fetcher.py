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


def fetch_all(path):
    with open(path, mode='rt', encoding='utf-8') as f:
        result = ''.join(f.readlines()).replace('\"', '').replace("v:", '"v":').replace("c:", '"c":').replace("'", '"') \
            .replace(',,', ',').replace('<a', '"<a').replace('</a>', '</a>"').replace('href="', "href='") \
            .replace('[uid]"', "[uid]'").replace('target="', "target='").replace('">', "'>")

    print(result)
    data = json.loads(result)
    print(len(data['rows']))


if __name__ == '__main__':
    fetch_all('/home/lucasx/Desktop/all_data.json')
    # trans_json('./data.json')
    """
    fetch(
        'http://arabidopsis.gmi.oeaw.ac.at:5000/DisplayResultsGene/fetchTopCandidateGenesFromOneResultOneGeneList?type_id'
        '=56&phenotype_method_id=11&analysis_method_id=2&list_type_id=0&max_rank=50')
    """
