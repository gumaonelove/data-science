from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from requests import get
from tqdm import tqdm

MAX_WORKERS = 1

details = pd.read_csv('data/details.csv')
vins = np.array_split(details['vin'].to_list(), MAX_WORKERS)
url = 'https://vpic.nhtsa.dot.gov/api/vehicles/decodevinextended/'


def parse_json_from_nhtsa(response) -> dict:
    '''Получение информации из json, игнорируя пустые фичи'''
    info = {}
    results = response['Results']
    for result in results:
        if result['Value']:
            info[result['Variable']] = result['Value']
    return info


def get_url(vins):
    global url
    vin_info_from_nhtsa = []
    for vin in vins:
        response = get(url=url + vin, params={'format': 'json'})
        if response.status_code == 200:
            info = parse_json_from_nhtsa(response.json())
            info['vin'] = vin
            vin_info_from_nhtsa.append(info)
        else:
            print(response.status_code)
            print(response)
            return
    return vin_info_from_nhtsa


all_info = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    all_info.extend(tqdm(pool.map(get_url, vins), total=len(vins)))

df_vin_info_from_nhtsa = pd.DataFrame(all_info)
df_vin_info_from_nhtsa.to_csv('df_vin_info_from_nhtsa.csv')
