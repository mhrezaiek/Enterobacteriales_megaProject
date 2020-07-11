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
def snp_collector(input, output):
    output_path = output
    files = os.listdir()
    genome_id_creator()
    for f in files:
        path = f + "/snps.csv"
        order = "cp " + path + " ../SNPs/"
        os.system()


if __name__ == "__main__":
    genome_id = genome_id_creator("Escherichia_just_gene_numbers_unique.csv")
    print(genome_id[1][1])
