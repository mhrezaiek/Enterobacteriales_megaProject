library(data.table)
library(ggplot2)
library(tidyverse)
library("dplyr")
data = fread("Desktop/Streptococcus_snps_table.csv")
label = fread("~/Documents/PycharmProjects/Escherichia_project/Streptococcus/Dataset /Perdrug/vancomycin_final.csv")
colnames(data)[1] <- "id"
colnames(label)[1] <- "genome_id"

#dup <- which(duplicated(colnames(data)))
#data = data[, -c(52, 53, 54, 55,56, 57,58,59,72,73,74,77,78,79,80,81,82,83,84,85,86,87,88 )]

data = merge(data,label, by.x = "id", by.y = "genome_id", all.x = TRUE)
data[is.na(data)] <- -1
data$vancomycin_Value<-NULL



write.csv(data, "~/Documents/PycharmProjects/Escherichia_project/Streptococcus/Dataset /Streptococcus_10drugs.csv" )
