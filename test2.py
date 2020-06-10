# spark-yarn.py
import happybase
import glob
import pandas as pd

connection = happybase.Connection()
table = connection.table('books')

connection.create_table(
    'neuro_test',
    {'data': dict(),
     'metadata': dict(),  # use defaults
    }
)

table = connection.table('neuro_test')


list_files = sorted(glob.glob('/data/test/roi/*.csv'))
for file_name in list_files:
    data = pd.read_csv(file_name)
    data = data.to_json().encode('utf-8')
    person = file_name.split('/')[-1].split('.')[0]
    person_number = file_name.split('/')[-1].split('_')[0]
    row = 'HCP_'+person
    row = row.encode('utf-8')
    table.put(row, {b'data:person': person_number.encode('utf-8'),
                               b'data:dataset': b'HCP',
                               b'data:atlas': b'harvard-oxford',
                               b'data:roi': data})

# row = table.row(b'Godel, Escher, Bach')
# print(row[b'analytics:views'])  # prints 'value1'

# for key, data in table.rows([b'row-key-1', b'row-key-2']):
#     print(key, data)  # prints row key and data for each row

# for key, data in table.scan(row_prefix=b'row'):
#     print(key, data)  # prints 'value1' and 'value2'

# row = table.delete(b'row-key')
