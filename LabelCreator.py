import pandas as pd
import csv
import os
import math


def genome_id_creator(path):
    gene_list = path
    genome_id = []
    with open(gene_list) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            genome_id.append(row)
    return genome_id


def Label_creator(raw_table_path,Isolates_table_path):
    isolates_table = genome_id_creator(Isolates_table_path)
    raw_table = genome_id_creator(raw_table_path)
    drugs = ["amoxicillin", "ampicillin", "aztreonam", "cefepime", "cefotaxime", "cefoxitin","ceftazidime", "cefuroxime",
                     "ciprofloxacin","gentamicin", "piperacillin","tobramycin", "trimethoprim","tigecycline","imipenem",
                     "meropenem","amikacin","ertapenem","ceftriaxone","tetracycline","cefalotin"]

    for i in range(len(isolates_table)-1):
        phenotype = ""
        measurement_value = ""
        gene_id = isolates_table[i + 1][0]
        antibiotic = str(isolates_table[i+1][3])
        print(str(i) + str(gene_id) + str(antibiotic))
        if isolates_table[i+1][4] != "":
            phenotype = str(isolates_table[i + 1][4])
            if phenotype == "Resistant":
                phenotype = 1
            elif phenotype == "Susceptible":
                phenotype = 0
            elif phenotype == "Intermediate":
                phenotype = 1
        if isolates_table[i+1][7] != "":
            measurement_value = float(isolates_table[i + 1][7])
        for gene in range(len(raw_table)):
            test_gene = raw_table[gene][0]
            if gene_id == test_gene:
                for drug in range(len(drugs)):
                    if antibiotic == drugs[drug]:
                        raw_table[gene][drug+1] = phenotype
                        raw_table[gene][drug+1+21] = measurement_value
    dataset = pd.DataFrame(raw_table)
    dataset.to_csv("test1.csv", index=False)


if __name__ == "__main__":
    Label_creator("raw_dataset_1.csv", "Escherichia_isolates.csv")



