import csv
import os
import pandas as pd

#### 1 stands for Resistance and 0 indicates suseptable


def genome_id_creator(path):
    gene_list = path
    genome_id = []
    with open(gene_list) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            genome_id.append(row)
    return genome_id


def value_convertor(lable_table_path):
    final_dataset = []
    dataset = genome_id_creator(lable_table_path)
    for i in range(len(dataset)):
        if i ==0:
            continue
        else:
            if dataset[i][2] != "" and float(dataset[i][2])!=-2:
                if float(dataset[i][2]) <=8and float(dataset[i][2])>0 and dataset[0][1] == "amoxicillin" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) <=8and float(dataset[i][2])>0 and dataset[0][1] == "ampicillin" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) <=8and float(dataset[i][2])>0 and dataset[0][1] == "aztreonam" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) <=4and float(dataset[i][2])>0 and dataset[0][1] == "cefepime" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) <=1and float(dataset[i][2])>0 and dataset[0][1] == "cefepime" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >=2  and dataset[0][1] == "cefepime" :
                    dataset[i][1] = 1
                elif float(dataset[i][2]) <=8and float(dataset[i][2])>0 and dataset[0][1] == "cefoxitin" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >=32  and dataset[0][1] == "cefoxitin" :
                    dataset[i][1] = 1
                elif float(dataset[i][2]) <=1 and float(dataset[i][2])>0 and dataset[0][1] == "ceftazidime" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >4  and dataset[0][1] == "ceftazidime" :
                    dataset[i][1] = 1
                elif float(dataset[i][2]) <=0.001 and float(dataset[i][2])>0 and dataset[0][1] == "cefuroxime" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >8  and dataset[0][1] == "cefuroxime" :
                    dataset[i][1] = 1
                elif float(dataset[i][2]) <=0.25 and float(dataset[i][2])>0 and dataset[0][1] == "ciprofloxacin" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >0.5  and dataset[0][1] == "ciprofloxacin" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=2 and float(dataset[i][2])>0 and dataset[0][1] == "gentamicin" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >2  and dataset[0][1] == "gentamicin" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=8 and float(dataset[i][2])>0 and dataset[0][1] == "piperacillin" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >16  and dataset[0][1] == "piperacillin" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=2 and float(dataset[i][2])>0 and dataset[0][1] == "tobramycin" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >2  and dataset[0][1] == "tobramycin" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=0.5 and float(dataset[i][2])>0 and dataset[0][1] == "tigecycline" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >0.5  and dataset[0][1] == "tigecycline" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=2 and float(dataset[i][2])>0 and dataset[0][1] == "imipenem" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >4  and dataset[0][1] == "imipenem" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=2 and float(dataset[i][2])>0 and dataset[0][1] == "meropenem" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >8  and dataset[0][1] == "meropenem" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=8 and float(dataset[i][2])>0 and dataset[0][1] == "amikacin" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >8  and dataset[0][1] == "amikacin" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=0.5 and float(dataset[i][2])>0 and dataset[0][1] == "ertapenem" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >0.5  and dataset[0][1] == "ertapenem" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=1 and float(dataset[i][2])>0 and dataset[0][1] == "ceftriaxone" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >2  and dataset[0][1] == "ceftriaxone" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=0.5 and float(dataset[i][2])>0 and dataset[0][1] == "tetracycline" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >0.5  and dataset[0][1] == "tetracycline" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=16 and float(dataset[i][2])>0 and dataset[0][1] == "cefalotin" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >16  and dataset[0][1] == "cefalotin" :
                    dataset[i][1] = 1

                else:
                    dataset[i][1] = 1


    final_dataset = pd.DataFrame(dataset)

    final_dataset.to_csv("datasets_perdrug/trimethoprim_final.csv", index=False, header=False)

if __name__ == "__main__":
    value_convertor("datasets_perdrug/trimethoprim.csv")

