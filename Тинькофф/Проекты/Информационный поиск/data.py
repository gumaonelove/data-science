import pandas as pd
import numpy as np


def get_data(low: bool = True) -> np.array:
    df: pd.DataFrame = pd.read_csv('clear_data.csv', index_col='cord_uid')
    if low: df = df.head()
    df['authors']: pd.Series = df['authors'].str.replace(';', ',')
    df['authors']: pd.Series = df['authors'].str.strip()
    np_ar: np.array = df.to_numpy()

    return np_ar


class Document:
    def __init__(self, title: str, authors: str):
        '''Можете здесь какие-нибудь свои поля подобавлять'''
        self.title: str = title
        self.authors: str = authors

    def format(self, query) -> list:
        '''Возвращает пару тайтл-текст, отформатированную под запрос'''
        return [self.title[:100] + ' ...', self.authors[:100] + ' ...']

    def __str__(self):
        return f'{self.title} | {self.authors}'