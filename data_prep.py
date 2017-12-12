import pandas as pd

colNames = ["Xmv_1","Xmv_2","Xmv_3","Xmv_4","Xmv_5","Xmv_6","Xmv_7","Xmv_8","Xmv_9","Xmv_10","Xmv_11",
                                     "Xmv_12","XMEAS_1", "XMEAS_2", "XMEAS_3", "XMEAS_4", "XMEAS_5", "XMEAS_6", "XMEAS_7",
                                     "XMEAS_8", "XMEAS_9", "XMEAS_10",
                                     "XMEAS_11", "XMEAS_12", "XMEAS_13", "XMEAS_14", "XMEAS_15", "XMEAS_16", "XMEAS_17",
                                     "XMEAS_18", "XMEAS_19",
                                     "XMEAS_20", "XMEAS_21", "XMEAS_22", "XMEAS_23", "XMEAS_24", "XMEAS_25", "XMEAS_26",
                                     "XMEAS_27", "XMEAS_28",
                                     "XMEAS_29", "XMEAS_30", "XMEAS_31", "XMEAS_32", "XMEAS_33", "XMEAS_34", "XMEAS_35",
                                     "XMEAS_36", "XMEAS_37",
                                     "XMEAS_38", "XMEAS_39", "XMEAS_40", "XMEAS_41"]


def import_data(data_path,condition,mode,fault_id,step,final):


    list_aux = []

    for x in range(step, final + step, step):
        df_aux = pd.read_csv(data_path + condition + mode+ '_ID_' + str(fault_id) + '_'+str(x) + '.csv',
                             names=colNames)

        list_aux.insert(len(list_aux), df_aux)

    df = pd.concat(list_aux, ignore_index=True)

    return df




if __name__ == "__main__":
    print("Main")
