setwd("/home/msmith/kaggle/lung/luna16/")

saveDir = "../writeUp/"

library(ggplot2)
cands = read.csv("candidates_V2.csv")
anno = read.csv("annotations.csv")

table(cands$class)

TP = cands[cands$class==1,]
FP = cands[cands$class==0,]
diam_summary = summary(anno$diameter_mm)
ggplot(data = anno,aes(anno$diameter_mm)) + geom_histogram(binwidth=1, col="Grey", fill="Black") +
  xlab("Nodule diameter (mm)") + 
  ylab("Frequency") +
  ggtitle("Distribution of nodule annotation diameter")
ggsave(paste(saveDir,"noduleDiameter.jpg",sep=""))
  #geom_vline(xintercept=diam_summary["Median"])





