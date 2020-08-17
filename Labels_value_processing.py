import csv
import os
import pandas as pd


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
                if float(dataset[i][2]) <8and float(dataset[i][2])>0:
                    dataset[i][1] = 0
                else:
                    dataset[i][1] = 1

    final_dataset = pd.DataFrame(dataset)

    final_dataset.to_csv("datasets_perdrug/trimethoprim_1.csv", index=False)

if __name__ == "__main__":
    value_convertor("datasets_perdrug/trimethoprim.csv")

