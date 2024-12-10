rm(list=ls())
setwd("/Users/dominikheider/Desktop/GNUS/")
library(DMwR)
library(ROCR)
library(sfsmisc)
library(randomForest)
library(kernlab)
library(smotefamily)

### setting parameters
mean.frac = 0.001
sd.frac = 0.001

### Choose dataset
data.name = "wdbc"


##################

### breast cancer diagnostics
if(data.name == "wdbc"){
  file = "./results_wdbc/Noise_"
  
  i=126
  set.seed(i)
  breast_diagnostic <- read.csv(header=F,"./data/wdbc.data.txt")
  breast_diagnostic[breast_diagnostic[,"V2"]=="B",1] <- 0
  breast_diagnostic[breast_diagnostic[,"V2"]=="M",1] <- 1
  breast_diagnostic <- breast_diagnostic[, -c(2)]
  cases <- breast_diagnostic[breast_diagnostic[,"V1"]==1,]
  controls <- breast_diagnostic[breast_diagnostic[,"V1"]==0,]
}

### breast cancer prognostics
if(data.name == "wpbc"){
  file = "./results_wpbc/Noise_"
  
  i=127
  set.seed(i)
  breast_prognostic <- read.csv(header=F,"./data/wpbc.data.txt")
  breast_prognostic[breast_prognostic[,"V2"]=="N",1] <- 0 #class 0
  breast_prognostic[breast_prognostic[,"V2"]=="R",1] <- 1 #class 1
  #breast_prognostic <- breast_prognostic[!(breast_prognostic$V35="?"),]
  breast_prognostic <- breast_prognostic[, -c(2,35)]
  cases <- breast_prognostic[breast_prognostic[,"V1"]==1,]
  controls <- breast_prognostic[breast_prognostic[,"V1"]==0,]

}

### cervical cancer
if(data.name == "cervical"){
  file = "./results_cervical/Noise_"
  i=128
  set.seed(i)
  cervical_cancer <- read.csv("./data/risk_factors_cervical_cancer_clean.csv", sep=";", header=T)
  NAs = c()
  for(j in 1:length(cervical_cancer$Age)){
    hit = which(is.na(cervical_cancer[j,]))
    if(length(hit)>0){
      NAs = c(NAs, j)
    }
  }
  cervical_cancer = cervical_cancer[-c(NAs),]
  V1 = cervical_cancer$Dx_Cancer
  cervical_cancer = cervical_cancer[,-8]
  cervical_cancer = cbind(V1, cervical_cancer)
  cases <- cervical_cancer[cervical_cancer[,"V1"]==1,]
  controls <- cervical_cancer[cervical_cancer[,"V1"]==0,]
}

### fertility
if(data.name == "fertility"){
  file = "./results_fertility/Noise_"
  
  i=129
  set.seed(i)
  fertility = read.csv("./data/fertility_Diagnosis.txt", header=F)
  colnames(fertility) = c("Season", "Age", "diseases", "Accident", "Surgical", "fevers", "alcohol", 
                          "smoking", "sitting", "Output")
  V1 = as.character(fertility$Output)
  fertility = fertility[,-10]
  fertility = as.data.frame(fertility)
  fertility = cbind(V1, fertility)
  
  fertility$V1 = as.character(fertility$V1)
  
  fertility[fertility[,"V1"]=="N",1] <- 0 #class 0
  fertility[fertility[,"V1"]=="O",1] <- 1 #class 1
  cases <- fertility[fertility[,"V1"]==1,]
  controls <- fertility[fertility[,"V1"]==0,]
}
#######

### sobar data set / cervix
if(data.name == "sobar"){
  file = "./results_sobar/Noise_"
  i=130
  set.seed(i)
  
  cervix = read.csv("./data/sobar-72.csv", header=T)
  V1 = as.character(cervix$ca_cervix)
  cervix = cervix[,-20]
  cervix = as.data.frame(cervix)
  cervix = cbind(V1, cervix)
  cervix$V1 = as.character(cervix$V1)
  cases <- cervix[cervix[,"V1"]==1,]
  controls <- cervix[cervix[,"V1"]==0,]
}

### Haberman data set / breast cancer survival
if(data.name == "haberman"){
  file = "./results_haberman/Noise_"
  i=131
  set.seed(i)
  
  haberman = read.csv("./data/haberman.data", header=F)
  colnames(haberman) = c("V2", "V3", "V4", "V1")
  V1 = as.numeric(haberman$V1)
  V1 = V1-1
  haberman = haberman[,-4]
  haberman = as.data.frame(haberman)
  haberman = cbind(V1, haberman)
  haberman$V1 = as.character(haberman$V1)
  cases <- haberman[haberman[,"V1"]==1,]
  controls <- haberman[haberman[,"V1"]==0,]
}

### Drug consumption / only heroin
if(data.name == "drug"){
  file = "./results_drug/Noise_"
  
  i=132
  set.seed(i)
  drug = read.csv("./data/drug_consumption.csv", header=F)
  #V3 = gender, f=0.48246, m=0.-48246
  drug <- drug[drug[,"V3"]==0.48246,] #select only female
  drug = drug[,-c(1,3)]                    #delete id and gender
  ## select drug, e.g., V24=Heroin
  ## CL0 Never Used 1627 86.31% 1021 54.16% 1605 85.15% 1490 79.05%
  ## CL1 Used over a Decade Ago 67 3.55% 113 5.99% 68 3.61% 45 2.39%
  ##  CL2 Used in Last Decade 112 5.94% 234 12.41% 94 4.99% 142 7.53%
  ##  CL3 Used in Last Year 59 3.13% 277 14.69% 65 3.45% 129 6.84%
  ##  CL4 Used in Last Month 9 0.48% 156 8.28% 24 1.27% 42 2.23%
  ##  CL5 Used in Last Week 9 0.48% 63 3.34% 16 0.85% 33 1.75%
  ##  CL6 Used in Last Day 2 0.11% 21 1.11% 13 0.69% 4 0.21% 
  
  V1 = as.character(drug$V24)
  for(j in 1:length(V1)){
    if(V1[j]=="CL0"){
      V1[j] = 0
    }else{
      V1[j] = 1
    }
  }
  
  drug = drug[,1:11]
  drug = as.data.frame(drug)
  drug = cbind(V1, drug)
  drug$V1 = as.character(drug$V1)
  cases <- drug[drug[,"V1"]==1,]
  controls <- drug[drug[,"V1"]==0,]
}

### HCV
if(data.name == "hcv"){
  file = "./results_hcv/Noise_"
  i=133
  set.seed(i)
  hcv = read.csv("./data/hcvdat0.csv", header=T)
  hcv = as.data.frame(hcv)
  hcv$Sex = as.character(hcv$Sex)
  ### make gender m=0, f=1
  hcv[hcv[,"Sex"]=="m",4] <- 0 #m=0
  hcv[hcv[,"Sex"]=="f",4] <- 1 #m=0
  hcv$Sex = as.numeric(hcv$Sex)
  controls = hcv[1:533,]
  cases = hcv[541:564,]
  hcv = rbind(cases, controls)
  V1 = as.character(c(rep(1,length(cases[,1])), rep(0, length(controls[,1]))))
  hcv = hcv[,-c(1,2)]
  hcv = as.data.frame(hcv)
  hcv = cbind(V1, hcv)
  hcv$V1 = as.character(hcv$V1)
  
  NAs = c()
  for(j in 1:length(hcv$V1)){
    hit = which(is.na(hcv[j,]))
    if(length(hit)>0){
      NAs = c(NAs, j)
    }
  }
  hcv = hcv[-c(NAs),]
  
  cases <- hcv[hcv[,"V1"]==1,]
  controls <- hcv[hcv[,"V1"]==0,]  
}

### NAFLD vs AFLD
if(data.name == "nafld"){
  file = "./results_nafld/Noise_"
  
  i=134
  set.seed(i)
  nafld <- read.csv("./data/NAFLD_AFL_clean.csv", header=T, sep=";", dec = ",")
  V1 = nafld[,1]
  nafld = nafld[,-1]
  nafld = as.data.frame(nafld)
  nafld = cbind(V1, nafld)
  
  nafld$V1 = as.character(nafld$V1)
  cases <- nafld[nafld[,"V1"]==1,]
  controls <- nafld[nafld[,"V1"]==0,]
}

### CTG prenatal
if(data.name == "ctg"){
  file = "./results_ctg/Noise_"
  
  #####  CTG prenatal
  i=135
  set.seed(i)
  ctg <- read.csv("./data/CTG_new.csv", header=T, sep=";", dec = ",")
  
  normal = ctg[which(ctg[,23]==1),]
  pathologic = ctg[which(ctg[,23]==3),]
  
  ctg = rbind(pathologic, normal)
  
  V1 = as.numeric(ctg[,"NSP"]) #1=normal, 2=susceptible, 3=pathologic, we take only 1 and 3
  V1[which(V1==1)] = 0
  V1[which(V1==3)] = 1
  
  ctg = ctg[,-23]
  ctg = as.data.frame(ctg)
  ctg = cbind(V1, ctg)
  
  cases <- ctg[ctg[,"V1"]==1,]
  controls <- ctg[ctg[,"V1"]==0,]
}


