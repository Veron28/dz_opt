from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

from typing import List


@dataclass
class Crews:
    vvl: float
    mvl: float
    sng: float


class FlyData:

    def __init__(self, path: str):

        self.df = pd.read_excel(path)
        self.df['День месяца'] = self.df['День месяца'].astype('str')

        self.number_of_crew = {
            1: Crews(vvl=125.54, mvl=119.88, sng=122.03),
            2: Crews(vvl=122.84, mvl=117.30, sng=119.41),
            3: Crews(vvl=129.97, mvl=124.10, sng=126.33),
            4: Crews(vvl=127.26, mvl=121.51, sng=123.69),
            5: Crews(vvl=123.32, mvl=117.76, sng=119.87),
            6: Crews(vvl=130.64, mvl=124.75, sng=126.99)
        }

        self.df_links, self.numbers = self.init_data_links(pd.get_dummies(
            self.df[['День месяца', 'Назначение', 'Тип судна', 'Тип связи']]))

        self.link_criteria1 = self.df_links[['Время полета_ВВЛ', 'Время полета_МВЛ', 'Время полета_СНГ']]
        self.link_criteria2 = self.df_links[
            ['Время полета(Ночной)_ВВЛ', 'Время полета(Ночной)_МВЛ', 'Время полета(Ночной)_СНГ']]
        self.link_criteria3 = self.df_links[['Экипаж_ВВЛ', 'Экипаж_МВЛ', 'Экипаж_СНГ']]
        self.link_criteria4 = self.df_links[['Ночной полет']]
        self.link_criteria5 = self.df_links[[f'День месяца_{i}' for i in range(1, 32)]]
        self.link_criteria6 = self.df_links[list(filter(lambda x: x.count('Тип судна_') != 0, self.df_links.columns))]
        self.link_criteria7 = self.df_links[list(filter(lambda x: x.count('Назначение_') != 0, self.df_links.columns))]

        self.shares = self.normalize_crews(self.number_of_crew)

        self.ideal = self.ideal_count()
        self.weight = self.to_weigh()
        self.sigma = self.to_sigma()

    def init_data_links(self, df_new: pd.DataFrame) -> pd.DataFrame:

        self._columns = list(df_new.columns)

        df_new['Время полета в часах'] = pd.to_datetime(self.df['Налет']).apply(
            lambda x: x.hour + (x.minute / 60)) * self.df['Экипаж']

        df_new = df_new.join(pd.get_dummies(self.df['Тип связи'], prefix='Время полета'))

        df_new['Время полета_ВВЛ'] = df_new['Время полета_ВВЛ'] * df_new['Время полета в часах']
        df_new['Время полета_МВЛ'] = df_new['Время полета_МВЛ'] * df_new['Время полета в часах']
        df_new['Время полета_СНГ'] = df_new['Время полета_СНГ'] * df_new['Время полета в часах']

        df_new['Ночной полет'] = (pd.to_datetime(self.df['Время вылета']).apply(
            lambda x: not ((x.hour >= 6) & ((x.hour < 22) or ((x.hour == 22) & (x.minute == 0))))))

        df_new = df_new.join(pd.get_dummies(self.df['Тип связи'], prefix='Время полета(Ночной)'))

        df_new['Время полета(Ночной)_ВВЛ'] = df_new['Время полета(Ночной)_ВВЛ'] * df_new['Ночной полет'] \
                                             * df_new['Время полета в часах']
        df_new['Время полета(Ночной)_МВЛ'] = df_new['Время полета(Ночной)_МВЛ'] * df_new['Ночной полет'] \
                                             * df_new['Время полета в часах']
        df_new['Время полета(Ночной)_СНГ'] = df_new['Время полета(Ночной)_СНГ'] * df_new['Ночной полет'] \
                                             * df_new['Время полета в часах']

        df_new = df_new.join(pd.get_dummies(self.df['Тип связи'], prefix='Экипаж'))

        df_new['Экипаж_ВВЛ'] = df_new['Экипаж_ВВЛ'] * self.df['Экипаж']
        df_new['Экипаж_МВЛ'] = df_new['Экипаж_МВЛ'] * self.df['Экипаж']
        df_new['Экипаж_СНГ'] = df_new['Экипаж_СНГ'] * self.df['Экипаж']

        self._columns.append('Ночной полет')
        self._columns = list(filter(lambda x: x.count('Тип связи') == 0, self._columns))

        return df_new, (df_new[self._columns].sum(axis=0)) / len(self.number_of_crew)

    def normalize_crews(self, number_of_crew: List[Crews]) -> np.array:

        h = np.array([list(asdict(number_of_crew[i]).values()) for i in range(1, 7)]).T

        for i in range(len(h)):
            h[i] = h[i] / sum(h[i])

        return h

    def ideal_count(self, ):

        shares3 = np.append(self.shares, self.shares, 0)
        shares3 = np.append(shares3, self.shares, 0)

        link_creteria1_3 = self.link_criteria1.join(self.link_criteria2).join(self.link_criteria3)

        ideal_shares = pd.DataFrame((shares3 * np.array([link_creteria1_3.sum(axis=0)]).T).T,
                                    columns=link_creteria1_3.columns)

        ideal_numbers = pd.DataFrame(self.numbers).T

        for i in range(len(self.number_of_crew) - 1):
            ideal_numbers = pd.concat([ideal_numbers, pd.DataFrame(ideal_numbers.iloc[-1]).T], ignore_index=True)

        return ideal_numbers.join(ideal_shares)

    def to_weigh(self, ):

        df_weight = None

        for criteria, const in [(self.link_criteria1, 14), (self.link_criteria2, 10),
                                (self.link_criteria3, 5), (self.link_criteria4, 2),
                                (self.link_criteria5, 3), (self.link_criteria6, 0.5),
                                (self.link_criteria7, 1)]:
            if df_weight is not None:
                df_weight = df_weight.join(pd.DataFrame(np.full(len(criteria.iloc[0]), const), index=list(criteria.columns)).T)
            else:
                df_weight = pd.DataFrame(np.full(len(criteria.iloc[0]), const), index=list(criteria.columns)).T

        return df_weight / df_weight.iloc[0].sum()

    def to_sigma(self):

        sigma = np.zeros((len(self.df), len(self.ideal.iloc[0]), len(self.number_of_crew))) + 1

        for n, i in self.df_links[['Время полета_ВВЛ', 'Время полета_МВЛ',
                                   'Время полета_СНГ']].iterrows():
            for k in range(1, len(self.number_of_crew) + 1):
                for l in range(3):
                    if l == 0:
                        loc = i.to_list()[0] / self.number_of_crew[k].vvl
                        sigma[n, l + 6, k - 1] = 1 / self.number_of_crew[k].vvl
                    elif l == 1:
                        loc = i.to_list()[1] / self.number_of_crew[k].mvl
                        sigma[n, l + 6, k - 1] = 1 / self.number_of_crew[k].mvl
                    else:
                        loc = i.to_list()[2] / self.number_of_crew[k].sng
                        sigma[n, l + 6, k - 1] = 1 / self.number_of_crew[k].mvl

                    sigma[n, l, k - 1] = loc
                    sigma[n, l + 3, k - 1] = loc

        return sigma

if __name__ == '__main__':

    data = FlyData('/Users/ivankusenko/Downloads/ÐÑÑÐ¾Ð´Ð½ÑÐµ Ð´Ð°Ð½Ð½ÑÐµ 2/ID.xls')

    print(data.sigma.shape)