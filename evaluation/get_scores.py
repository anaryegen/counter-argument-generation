import pandas as pd
import sys

def get_data(d, column_name,row):
    data = pd.read_csv(d)
    row = int(row)
    print(row)
    print('Opposition: ', data[column_name+'_opposition_exp'][row])
    print('Relatedness: ', data[column_name+'_relatedness_exp'][row])
    print('Specificity: ', data[column_name+'_specificity_exp'][row])
    print('Factuality: ', data[column_name+'_factuality_exp'][row])
    print('Fluency: ', data[column_name+'_fluency_exp'][row])
print('args1: ', sys.argv)
get_data(sys.argv[1], sys.argv[2], sys.argv[3])
#print('args1: ', sys.argv)
