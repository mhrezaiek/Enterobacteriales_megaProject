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
            if dataset[i][2] != "" and float(dataset[i][2])!=-2 and float(dataset[i][2]) == -2:
                if float(dataset[i][2]) <=2and float(dataset[i][2])>0 and dataset[0][1] == "trimethoprim/sulfamethoxazole" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >=4  and dataset[0][1] == "trimethoprim/sulfamethoxazole" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=0and float(dataset[i][2])>0 and dataset[0][1] == "erythromycin" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >=16  and dataset[0][1] == "erythromycin" :
                    dataset[i][1] = 1
                elif float(dataset[i][2]) <=0.25and float(dataset[i][2])>0 and dataset[0][1] == "penicillin" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >=0.25  and dataset[0][1] == "penicillin" :
                    dataset[i][1] = 1
                elif float(dataset[i][2]) <=8 and float(dataset[i][2])>0 and dataset[0][1] == "chloramphenicol" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >8  and dataset[0][1] == "chloramphenicol" :
                    dataset[i][1] = 1
                elif float(dataset[i][2]) <=1 and float(dataset[i][2])>0 and dataset[0][1] == "tetracycline" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >2  and dataset[0][1] == "tetracycline" :
                    dataset[i][1] = 1
                elif float(dataset[i][2]) <=0 and float(dataset[i][2])>0 and dataset[0][1] == "cotrimoxazole" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >0  and dataset[0][1] == "cotrimoxazole" :
                    dataset[i][1] = 1
                elif float(dataset[i][2]) <=0 and float(dataset[i][2])>0 and dataset[0][1] == "lactam" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >0  and dataset[0][1] == "lactam" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=0.5 and float(dataset[i][2])>0 and dataset[0][1] == "clindamycin" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >0.5  and dataset[0][1] == "clindamycin" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=2 and float(dataset[i][2])>0 and dataset[0][1] == "linezolid" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >2  and dataset[0][1] == "linezolid" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=2 and float(dataset[i][2])>0 and dataset[0][1] == "vancomycin" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >2  and dataset[0][1] == "vancomycin" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=0.25 and float(dataset[i][2])>0 and dataset[0][1] == "cefotaxime" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >0.25  and dataset[0][1] == "cefotaxime" :
                    dataset[i][1] = 1

                elif float(dataset[i][2]) <=0.001 and float(dataset[i][2])>0 and dataset[0][1] == "levofloxacin" :
                    dataset[i][1] = 0
                elif float(dataset[i][2]) >2  and dataset[0][1] == "levofloxacin" :
                    dataset[i][1] =1

                else:
                    dataset[i][1] = 1


    final_dataset = pd.DataFrame(dataset)

    final_dataset.to_csv("Dataset /Perdrug/vancomycin.csv", index=False, header=False)

if __name__ == "__main__":
    value_convertor("Dataset /Perdrug/vancomycin.csv")

