#load packages
library(HIMA)
library(boot)

#load data
df <- read.csv("/Users/yliu/Downloads/mediation_tst_shifted.csv")
#define statistics
#no confounder
fc_noconf_acme <- function(data,indices){
  d <- data[indices,]
  hima.fit <- hima(X = d$T, 
                   Y = d$Y, 
                   M = d[,1:616],
                   scale = FALSE,
                   verbose = FALSE) 
  himma_min <- hima.fit[hima.fit$Bonferroni.p==min(hima.fit$Bonferroni.p),]
  true_ACME_t1 <- -13.8063
  abs_error <- abs(himma_min$`alpha*beta`[1] - true_ACME_t1)
  return(abs_error)
}

set.seed(626)
bootcorr <- boot(df, fc_noconf_acme, R=10)
bootcorr


fc_noconf_ade <- function(data,indices){
  d <- data[indices,]
  hima.fit <- hima(X = d$T, 
                   Y = d$Y, 
                   M = d[,1:616],
                   scale = FALSE,
                   verbose = FALSE) 
  himma_min <- hima.fit[hima.fit$Bonferroni.p==min(hima.fit$Bonferroni.p),]
  true_ADE_t0 <- 0.6912
  abs_error <- abs(himma_min$`gamma`[1] - true_ADE_t0)
  return(abs_error)
}

set.seed(626)
bootcorr <- boot(df, fc_noconf_ade, R=10)
bootcorr



fc_noconf_ate <- function(data,indices){
  d <- data[indices,]
  hima.fit <- hima(X = d$T, 
                   Y = d$Y, 
                   M = d[,1:616],
                   scale = FALSE,
                   verbose = FALSE) 
  himma_min <- hima.fit[hima.fit$Bonferroni.p==min(hima.fit$Bonferroni.p),]
  true_ATE <- -13.1151
  predict_ATE <- himma_min$`gamma`[1] + himma_min$`alpha*beta`[1]
  abs_error <- abs(predict_ATE - true_ATE)
  return(abs_error)
}

set.seed(626)
bootcorr <- boot(df, fc_noconf_ate, R=10)
bootcorr

#with confounder
fc_acme <- function(data,indices){
  d <- data[indices,]
  hima.fit <- hima(X = d$T, 
                   Y = d$Y, 
                   M = d[,1:616],
                   COV.XM = d[, c("W","W")],
                   scale = FALSE,
                   verbose = FALSE) 
  himma_min <- hima.fit[hima.fit$Bonferroni.p==min(hima.fit$Bonferroni.p),]
  true_ACME_t1 <- 2.8335
  abs_error <- abs(himma_min$`alpha*beta`[1] - true_ACME_t1)
  return(abs_error)
}

set.seed(626)
bootcorr <- boot(df, fc_acme, R=10)
bootcorr

fc_ade <- function(data,indices){
  d <- data[indices,]
  hima.fit <- hima(X = d$T, 
                   Y = d$Y, 
                   M = d[,1:616],
                   COV.XM = d[, c("W","W")],
                   scale = FALSE,
                   verbose = FALSE) 
  himma_min <- hima.fit[hima.fit$Bonferroni.p==min(hima.fit$Bonferroni.p),]
  true_ADE_t0 <- 1.3039
  abs_error <- abs(himma_min$`gamma`[1] - true_ADE_t0)
  return(abs_error)
}


set.seed(626)
bootcorr <- boot(df, fc_ade, R=10)
bootcorr


fc_ate <- function(data,indices){
  d <- data[indices,]
  hima.fit <- hima(X = d$T, 
                   Y = d$Y, 
                   M = d[,1:616],
                   COV.XM = d[, c("W","W")],
                   scale = FALSE,
                   verbose = FALSE) 
  himma_min <- hima.fit[hima.fit$Bonferroni.p==min(hima.fit$Bonferroni.p),]
  true_ATE <- 4.1374
  abs_error <- abs(himma_min$`gamma`[1] + himma_min$`alpha*beta`[1] - true_ATE)
  return(abs_error)
}

set.seed(626)
bootcorr <- boot(df, fc_ate, R=10)
bootcorr
