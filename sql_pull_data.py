"""
Teste
"""

import matplotlib.pyplot as plt
import pandas as pd
import pyodbc as sql

server = "DESKTOP-2MP4P33"
db = "Teste"
UID = "Lais-WHart"
pswd = "LAISlab"

conn = sql.connect(
    'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + db + '; UID = ' + UID + '; PWD = ' + pswd + 'Trusted_Connection=yes')

df = pd.read_sql("SELECT * FROM dbo.TagHistorian;", conn)

plt.plot(pd.DataFrame(df, columns=['Lista_1_']))
plt.show(block=False)

print(df.head())
plt.show()
