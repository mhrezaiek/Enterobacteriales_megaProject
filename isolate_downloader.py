import os
import wget
import pandas as pd
import csv
import numpy as np


#file = pd.read_csv("Escherichia_just_gene_numbers.csv



genome_id = []
with open('Escherichia_just_gene_numbers_unique.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                genome_id.append(row)




files = []
files = os.listdir("../Escherichia/")
counter =0




for i in range(1625,len(genome_id)):
    strr = str(genome_id[i][1])+".fna"
    if strr in files:
        print(i)
    else:
          try:
              url = "ftp://ftp.patricbrc.org/genomes/"+str(genome_id[i][1])+"/"+str(genome_id[i][1])+".fna"
              wget.download(url)
              counter = +counter + 1
              if (counter%5 == 0):
                print(counter, flush=True)
          except:
              print("No such file id: "+str(genome_id[i][1]) )

