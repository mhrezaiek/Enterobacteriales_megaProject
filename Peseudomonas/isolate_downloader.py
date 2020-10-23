import os
import wget
import pandas as pd
import csv
import numpy as np


#file = pd.read_csv("Escherichia_just_gene_numbers.csv



def genome_id_creator(path):
    gene_list = path
    genome_id = []
    with open(gene_list) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            genome_id.append(row)
    return genome_id




files = []
files = os.listdir("../Escherichia/")
counter =0

genome_id = []
genome_id = genome_id_creator("datasets/Peseudomonas_uniqe_isolate_num.csv")

for i in range(2):
    temp = str(genome_id[i])
    temp = temp[2:len(temp)-2]
    strr = "287.782"# temp
    try:
        url = "ftp://ftp.patricbrc.org/genomes/"+strr+"/"+strr+".fna"
        print(strr)
        wget.download(url)
        counter = +counter + 1
        if (counter%5 == 0):
            print(counter, flush=True)
    except:
        print("No such file id: "+str(genome_id[i]) )

