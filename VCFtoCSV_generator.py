import pandas
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


### for this function, all non-folder files should be moved from the directory
def snp_collector(input,output,gene_name_path):
    output_path = output
    files = os.listdir(input)
    genome_id = genome_id_creator(gene_name_path)
    for f in files:
        if f != ".DS_Store":
            path = input + f + "/snps.csv"
            target_csv_name = str(genome_id[int(f)+1][1]) + ".csv"
            order = "cp " + path + " " + output_path + target_csv_name
            os.system(order)
            

if __name__ == "__main__":
    snp_collector("../files/","../SNPs/", "Escherichia_just_gene_numbers_unique.csv")
    print("Done!")
