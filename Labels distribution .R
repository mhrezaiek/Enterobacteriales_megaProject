
library(ggplot2)
data = fread("Desktop/Escherichia_isolates.csv")

table_antibiotic<-as.data.frame( table(data$antibiotic))
t <- table_antibiotic[order(table_antibiotic$Freq),]

t$Var1 <- factor(t$Var1, levels = t$Var1)
ggplot(t, aes(x=Var1, y = Freq)) + 
  geom_bar(stat="identity")+theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

