import os
import wget
import pandas as pd
import csv





def genome_id_creator(path):
    gene_list = path
    genome_id = []
    with open(gene_list) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            genome_id.append(row)
    return genome_id



def snp_generator(input, output, reference, isolate_list_path):
    genome_id = genome_id_creator(isolate_list_path)
    for i in range(len(genome_id)):
        try:
            reference = reference
            output = output + str(genome_id[i + 1][1])
            ctgs = input + str(genome_id[i + 1][1]) + ".fna"

            order = "snippy --outdir " + output + " --ref " + reference +" --ctgs " + ctgs
            os.system(order)
            if (i % 5 == 0):
                print(i, flush=True)
            print(order)
        except:
            print("No such file id: " + str(genome_id[i + 1]))



if __name__ == "__main__":
    snp_generator()