library(data.table)
data = fread("Desktop/Escherichia_isolates.csv")
data = as.data.frame(data)
gene_id = fread("Desktop/Escherichia_just_gene_numbers_unique.csv")
gene_id = as.data.frame(gene_id)
gene_id$V1<- NULL
matrix = matrix(-2, nrow = 2268, ncol = 13)
matrix = as.data.frame(matrix)

colnames(matrix)<- c("Amoxicillin", "Ampicillin", "Aztreonam", "Cefepime", "Cefotaxime", "Cefoxitin","Ceftazidime", "Cefuroxime",
                     "Ciprofloxacin","Gentamicin", "Piperacillin","Tobramycin", "trimethoprim")


raw_dataset = cbind(gene_id,matrix)
write.csv(raw_dataset, "Desktop/raw_dataset.csv")
