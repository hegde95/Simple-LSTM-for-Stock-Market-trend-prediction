# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 00:23:31 2018

@author: Shashank
"""

import quandl
import pandas as p
import matplotlib.pyplot as plt 


def dataCreator(security,start_date,stop_date):
    quandl.ApiConfig.api_key = 'RDBQCtURbE4ysNQ1i2-5'
    data = quandl.get_table('WIKI/PRICES', ticker = security, 
    
                            qopts = { 'columns': ['ticker', 'date', 'adj_high','adj_low'] }, 
                            date = { 'gte': start_date,'lte': stop_date}, 
                            paginate=True)
    high_prices = data.loc[:,'adj_high'].as_matrix()
    low_prices = data.loc[:,'adj_low'].as_matrix()
    mid_prices = (high_prices+low_prices)/2.0
    data['mid_prices'] = p.Series(mid_prices, index=data.index)
#    data.set_index('date')
    data=data.sort_index(ascending=0)  
#    data.set_index('date')
    plt.plot(data['mid_prices'])
    return data


def main():
    data=dataCreator('AAL','1981-1-1','2018-06-31')
    data.to_csv('data2.csv', sep=',')

if __name__ == '__main__':
    main()
