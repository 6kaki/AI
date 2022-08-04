import pandas as pd

name = ['한현길','김기환','권장현','우성종','이민영']
math = [87,82,74,95,88]
eng = [86,95,97,86,88]

df = pd.DataFrame({
    '이름':name,
    '수학':math,
    '영어':eng
})

df['total'] = df['수학'] + df['영어']

df.to_csv('test.csv',encoding='utf-8-sig',index=False)

df = pd.read_csv('./test.csv')

new_name = ['a','b','c','d','f']
for i in range(len(df)):
    df.이름[i] = new_name[i]

df.to_excel('./ndf.xlsx',encoding='utf-8-sig',index=False)

ndf = pd.read_excel('./ndf.xlsx')
print(ndf)



