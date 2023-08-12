import pyodbc
server = 'tcp:localhost'
database = 'vPICList_Lite'
username = 'sa'
password = 'reallyStrongPwd123'
odbc_driver='ODBC Driver 18 for SQL Server'

cnxn = pyodbc.connect(
    'DRIVER={'+odbc_driver+'};SERVER='+server+';DATABASE='+database+';ENCRYPT=yes;UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

print(cursor)