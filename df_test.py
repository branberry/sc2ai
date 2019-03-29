import pandas as pd

df = pd.DataFrame(columns=['episode', 'return'])

df = df.append({'episode' : 1, 'return' : 232}, ignore_index=True)

print(df)

df.to_csv("test.csv")