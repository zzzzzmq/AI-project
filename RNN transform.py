# import pandas as pd

# def get_formatted_data():
#     csv_file = './data/DMSC.csv'
#     df = pd.read_csv(csv_file)
#     df['Comment'] = df['Comment'].apply(lambda x: x.replace('\n', ' '))
#     df['Text'] = df['Star'].astype(str) + '	####	' + df['Comment']

#     with open('dmsc.txt', 'w', encoding='utf-8') as file:
#         file.write('\n'.join(df['Text']))

# get_formatted_data()

import pandas as pd

def get_formatted_data():
    csv_file = './data/DMSC.csv'
    df = pd.read_csv(csv_file)
    df = df.dropna()
    df['Comment'] = df['Comment'].apply(lambda x: x.replace('\n', ' '))
    # 过滤并替换Stars列的值
    df['Star'] = df['Star'].apply(lambda x: 0 if x == 1 or x == 2 or x==3 else 1 )
    df['Text'] = (df['Star'].astype(int)).astype(str) + '	####	' + df['Comment']
    with open('dmsc.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(df['Text']))

get_formatted_data()
