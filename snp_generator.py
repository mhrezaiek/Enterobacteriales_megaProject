import os
import wget
import pandas as pd
import csv



#file = pd.read_csv("Escherichia_just_gene_numbers.csv")



genome_id = []
with open('Escherichia_just_gene_numbers_unique.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            genome_id.append(row)




#len(genome_id)

counter =0 
for i in range(len(genome_id)):
            try:    
                output = "../Escherichia/" + "add" + str(genome_id[i+1][1])
                ctgs = "../Ecoli_added_seq/"  + str(genome_id[i+1][1]) + ".fna"

                order = "snippy --outdir " + output + " --ref CP028307.1[1..4709905].fa --ctgs " + ctgs 
                os.system(order)
                if (i%5 == 0):
                    print(i, flush=True)
                print(order)    

            except:
                print("No such file id: "+str(genome_id[i+1]) )


