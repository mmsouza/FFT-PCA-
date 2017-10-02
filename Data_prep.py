import pandas as pd

normal_data = pd.read_csv('C:/Users/matheus/Desktop/normal_output.csv',
                          names = ["XMEAS_1","XMEAS_2","XMEAS_3","XMEAS_4","XMEAS_5","XMEAS_6","XMEAS_7","XMEAS_8","XMEAS_9","XMEAS_10",
                                   "XMEAS_11","XMEAS_12","XMEAS_13","XMEAS_14","XMEAS_15","XMEAS_16","XMEAS_17","XMEAS_18","XMEAS_19",
                                   "XMEAS_20","XMEAS_21","XMEAS_22","XMEAS_23","XMEAS_24","XMEAS_25","XMEAS_26","XMEAS_27","XMEAS_28",
                                   "XMEAS_29","XMEAS_30","XMEAS_31","XMEAS_32","XMEAS_33","XMEAS_34","XMEAS_35","XMEAS_36","XMEAS_37",
                                   "XMEAS_38","XMEAS_39","XMEAS_40","XMEAS_41"])



#normal_Idv = pd.read_csv('C:/Users/matheus/Desktop/normal_idv_output.csv')

normal_data['normal']=1
normal_data['failure']=0


list_aux = []

for x in range(24, 696 + 24, 24):
    df_aux = pd.read_csv('C:/Users/matheus/Desktop/Fault_TEP/Fault1_' + str(x) + '.csv',names = ["XMEAS_1","XMEAS_2","XMEAS_3","XMEAS_4","XMEAS_5","XMEAS_6","XMEAS_7","XMEAS_8","XMEAS_9","XMEAS_10",
                                   "XMEAS_11","XMEAS_12","XMEAS_13","XMEAS_14","XMEAS_15","XMEAS_16","XMEAS_17","XMEAS_18","XMEAS_19",
                                   "XMEAS_20","XMEAS_21","XMEAS_22","XMEAS_23","XMEAS_24","XMEAS_25","XMEAS_26","XMEAS_27","XMEAS_28",
                                   "XMEAS_29","XMEAS_30","XMEAS_31","XMEAS_32","XMEAS_33","XMEAS_34","XMEAS_35","XMEAS_36","XMEAS_37",
                                   "XMEAS_38","XMEAS_39","XMEAS_40","XMEAS_41"])

    list_aux.insert(len(list_aux), df_aux)
    print('C:/Users/matheus/Desktop/Fault_TEP/Fault1' + str(x))

Fault1_df = pd.concat(list_aux, ignore_index=True)

Fault1_df['normal'] = 0
Fault1_df['failure'] = 1


print(Fault1_df.describe())
