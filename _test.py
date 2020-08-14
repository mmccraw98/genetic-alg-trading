import pickle

keys = ['997e1d1d020b431ea8c60be928ab60b4',
        '7508da6e080349959f07d22e27241a13',
        'f3c0ea885d994a97819ec7f422449b91',
        '5e4ec8c18468445a8d5dcaf53fbad42d',
        'f8f52c34988e470d9992e8efe2fee6be',
        '﻿3c0b20c63bfe4085b33c09d2aa4296c8',
        '9fe855c9adec44cfa4a9bb551e6059c0',
        'cf4c2296894944b7a437bdf26bf43fbe',
        '4310149921f44ec798d954e9fd953cfd']

with open('.\\keys\\news_api\\news_api_keys_list.pkl', 'wb') as f:
    pickle.dump(keys, f)

with open('.\\keys\\news_api\\news_api_keys_list.pkl', 'rb') as f:
    print(pickle.load(f))