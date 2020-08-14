import csv
import os


def genome_id_creator(path):
    gene_list = path
    genome_id = []
    with open(gene_list) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            genome_id.append(row)
    return genome_id


def value_convertor(lable_table_path):
    dataset = genome_id_creator(lable_table_path)

