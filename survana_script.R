#########################################
# Survival analysis of liquidity measures
#########################################
library(survival)

rm(list=ls())
resetPar <- function() {
  dev.new()
  op <- par(no.readonly = TRUE)
  dev.off()
  op
}

par(resetPar())
####################################################################################
# Load csv for Sept 6th and Sept 9th
df06_is <- read.csv("/Users/mh/Documents/CSML/Masterarbeit/Python/TED_06_is.csv")
df09_is <- read.csv("/Users/mh/Documents/CSML/Masterarbeit/Python/TED_09_is.csv")
df06_is <- subset(df06_is, select = "TED")
df09_is <- subset(df09_is, select = "TED")
df06_is["event"] <- 1
df09_is["event"] <- 1
df06_is["group"] <- 0
df09_is["group"] <- 1
df06_is <- df06_is[df06_is$TED >= 0,]
df09_is <- df09_is[df09_is$TED >= 0,]
df_is <- rbind(df06_is, df09_is)
cat("Inside Spread")
cat("dimension of Sept 6th dataset: ", dim(df06_is))
cat("dimension of Sept 9th dataset: ", dim(df09_is))
cat("dimension of full dataset: ", dim(df_is))

df06_xlm <- read.csv("/Users/mh/Documents/CSML/Masterarbeit/Python/TED_06_xlm.csv")
df09_xlm <- read.csv("/Users/mh/Documents/CSML/Masterarbeit/Python/TED_09_xlm.csv")
df06_xlm <- subset(df06_xlm, select = "TED")
df09_xlm <- subset(df09_xlm, select = "TED")
df06_xlm["event"] <- 1
df09_xlm["event"] <- 1
df06_xlm["group"] <- 0
df09_xlm["group"] <- 1
df06_xlm <- df06_xlm[df06_xlm$TED >= 0,]
df09_xlm <- df09_xlm[df09_xlm$TED >= 0,]
df_xlm <- rbind(df06_xlm, df09_xlm)
cat("XLM")
cat("dimension of Sept 6th dataset: ", dim(df06_xlm))
cat("dimension of Sept 9th dataset: ", dim(df09_xlm))
cat("dimension of full dataset: ", dim(df_xlm))

################################################
# Create a survival object from raw data
tedSurv_is <- Surv(df_is$TED, df_is$event)
tedSurv06_is <- Surv(df06_is$TED, df06_is$event) 
tedSurv09_is <- Surv(df09_is$TED, df09_is$event) 

tedSurv_xlm <- Surv(df_xlm$TED, df_xlm$event)
tedSurv06_xlm <- Surv(df06_xlm$TED, df06_xlm$event) 
tedSurv09_xlm <- Surv(df09_xlm$TED, df09_xlm$event) 
################################################
##### Fit Kaplan Mayer curve
# Spread
dev.off()
jpeg("/Users/mh/Documents/CSML/Masterarbeit/tex/img/KM_highres.jpg", width = 7, height =5, units = 'in', res = 500)
par(mfrow=c(1,2))

fit06_is <- survfit(tedSurv06_is~1, conf.int=0.95)
fit09_is <- survfit(tedSurv09_is~1, conf.int=0.95)
plot(fit06_is, main="Kaplan-Meier curves for Spread", xlab="[seconds]",
     ylab="Estimate of P(S > t)", 
     ylim=c(0,0.6), 
     cex.lab=1, 
     cex.axis=1, 
     cex.main=1, 
     cex.sub=1,
     lwd=1.5)
lines(fit09_is, col = 'red', lwd=1.5)
legend(5, 0.4,
       c("June 6th", "June 9th"), lty=c(1,1), lwd=c(2.5,2.5),
       col=c("black","red"),
       bty="n")
# XLM
fit06_xlm <- survfit(tedSurv06_xlm~1, conf.int=0.95)
fit09_xlm <- survfit(tedSurv09_xlm~1, conf.int=0.95)
plot(fit06_xlm, main="Kaplan-Meier curves for XLM", xlab="[seconds]",
     ylab="Estimate of P(S > t)", 
     ylim=c(0,0.6), 
     cex.lab=1, 
     cex.axis=1, 
     cex.main=1, 
     cex.sub=1,
     lwd=3)
lines(fit09_xlm, col = 'red', lwd=1.5)
legend(5, 0.4,
       c("June 6th", "June 9th"), lty=c(1,1), lwd=c(2.5,2.5),
       col=c("black","red"),
       bty="n")
dev.off()
################################################
##### Compare Kaplan Mayer curves
diff_is <- survdiff(tedSurv_is ~ df_is$group)
diff_xlm <- survdiff(tedSurv_xlm ~ df_xlm$group)
diff_is
diff_xlm

################################################
##### Accelerated Time Failure Model
### Weibull Model ###

############################################################################
### SURVIVAL TIMES
# SPREAD
# 2014-06-06
time.06.is <- fit06_is$time
surv.06.is <- fit06_is$surv
time.06.e3.is <- fit06_is$time[fit06_is$time > 1e-2] # times > 1 millisecond
surv.06.e3.is <- fit06_is$surv[fit06_is$time > 1e-2] # corresponding surv. probs.
# 2014-06-09
time.09.is <- fit09_is$time
surv.09.is <- fit09_is$surv
time.09.e3.is <- fit09_is$time[fit09_is$time > 1e-2] # times > 1 millisecond
surv.09.e3.is <- fit09_is$surv[fit09_is$time > 1e-2] # corresponding surv. probs.
# XLM
# 2014-06-06
time.06.xlm <- fit06_xlm$time
surv.06.xlm <- fit06_xlm$surv
time.06.e3.xlm <- fit06_xlm$time[fit06_xlm$time > 1e-2] # times > 1 millisecond
surv.06.e3.xlm <- fit06_xlm$surv[fit06_xlm$time > 1e-2] # corresponding surv. probs.
# 2014-06-09
time.09.xlm <- fit09_xlm$time
surv.09.xlm <- fit09_xlm$surv
time.09.e3.xlm <- fit09_xlm$time[fit09_xlm$time > 1e-2] # times > 1 millisecond
surv.09.e3.xlm <- fit09_xlm$surv[fit09_xlm$time > 1e-2] # corresponding surv. probs.

####################################
### CHECKING WEIBULL ASSUMPTIONS
par(mfrow=c(2,2))
par(mar=c(0.01,0.01,0.01,0.01), oma=c(1.2,1.2,0,0))
plot(log(time.06.is), log(-log(surv.06.is)), type='l',lwd=3,
     xaxt='n', ylab='Spread', xlab='',
     cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
lines(log(time.09.is), log(-log(surv.09.is)), lwd=3, col=2,
      cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)

plot(log(time.06.e3.is), log(-log(surv.06.e3.is)), type='l',lwd=3,
     yaxt='n', xaxt='n', ylab='', xlab='',
     cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
lines(log(time.09.e3.is), log(-log(surv.09.e3.is)), lwd=3, col=2,
      cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
legend(-5, 2,
       c("June 6th 2014", "June 9th 2014"), lty=c(1,1), lwd=c(5,5),
       col=c("black","red"),
       bty="n",
       cex=1.6)

plot(log(time.06.xlm), log(-log(surv.06.xlm)), type='l',lwd=3,
     ylab='log(-log(S(t)))', xlab='log(t)',
     cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
lines(log(time.09.xlm), log(-log(surv.09.xlm)), lwd=3, col=2,
      cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)

#par(mai=c(1.2,1.2,0.01,0.01))
plot(log(time.06.e3.xlm), log(-log(surv.06.e3.xlm)), type='l',lwd=3,
     yaxt='n', ylab='', xlab='log(t)',
     cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
lines(log(time.09.e3.xlm), log(-log(surv.09.e3.xlm)), lwd=3, col=2,
      cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)

title(xlab = "All durations vs. durations > 10e-3",
      ylab = "XLM vs. Spread",
      outer = TRUE, line = 2.5,
      cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)


##############################################################################
### LOAD FEATURES
df06 <- read.csv("/Users/mh/Documents/CSML/Masterarbeit/Python/df_feat06.csv")
df09 <- read.csv("/Users/mh/Documents/CSML/Masterarbeit/Python/df_feat09.csv")
df06 <- df06[df06$TED>=1e-3,]
df09 <- df09[df09$TED>=1e-3,]
df06$MO_ask <- as.factor(df06$MO_ask)
df06$MO_bid <- as.factor(df06$MO_bid)
df09$MO_ask <- as.factor(df09$MO_ask)
df09$MO_bid <- as.factor(df09$MO_bid)

# Standardize numeric features as there are significant level differences between days
#df06["AvgTED_m5_save"] = df06$AvgTED_m5
nums <- sapply(df06, is.numeric)
nums["TED"] <- FALSE
nums["AvgTED_m5"] <- FALSE
df06[,nums] <- scale(df06[,nums])
df09[,nums] <- scale(df09[,nums])

# Survival objects
df06["event"] <- 1
df09["event"] <- 1
tedSurv06.xlm.e3 <- Surv(df06$TED, df06$event)
tedSurv09.xlm.e3 <- Surv(df09$TED, df09$event)
# Fit model
fit06 <- survreg(tedSurv06.xlm.e3 ~ df06$OrderImb + 
                                    df06$Spread + 
                                    df06$XLM + 
                                    df06$MO_ask + 
                                    df06$MO_bid + 
                                    df06$LastTED + 
                                    df06$AvgTED, dist='weibull')
summary(fit06)

fit09 <- survreg(tedSurv09.xlm.e3 ~ df09$OrderImb + 
                                   df09$Spread + 
                                   df09$XLM + 
                                   df09$MO_ask + 
                                   df09$MO_bid + 
                                   df09$LastTED + 
                                   df09$AvgTED, dist='weibull')
summary(fit09)


#### Model diagnostics: Cox Snell residuals
# June 6th
# 1. Get survival times (df06$TED)
# 2. Get predictions (fit06)
# 3. Compute the log of the Cox-Snell residuals: ln(R_i) = (ln(T_i) - betaHat*x_i) / scaleHat
dev.off()
quartz()
jpeg("/Users/mh/Documents/CSML/Masterarbeit/tex/img/AFT_diagnostics.jpg", width = 5, height =4, units = 'in', res = 1000)
par(mfrow=c(1,2))
# JUNE 6th
par(mai=c(0.7,0.7,0.7,0.05))
log.cox.snell <- (log(df06$TED) - fit06$linear.predictors) / fit06$scale
coef <- fit06$coefficients
fi <- coef[names(coef) == "df06$AvgTED"] + coef[names(coef) == "(Intercept)"] + 
      fit06$scale * log.cox.snell

plot(df06$AvgTED, fi, main="June 6th", xaxt='n', yaxt='n', xlab='', ylab='')
mtext(side = 1, text = "Avg. last five TEDs", line = 0.7)
mtext(side = 2, text = "f(AvgTED)", line = 0.7)

reg.linear <- lm(fi ~ df06$AvgTED)
reg.nonlinear <- lm(fi ~ I((0.1+df06$AvgTED)^(-1)) + df06$AvgTED)
tmp <- data.frame(df06$AvgTED, fi, reg.nonlinear$fitted.values, reg.linear$fitted.values)
names(tmp) <- c("AvgTED", "fi", "reg.nl", "reg.lin")
tmp <- tmp[order(tmp$AvgTED), ]
lines(tmp$AvgTED, tmp$reg.lin, col="blue", lwd=3, xaxt='n', yaxt='n')
lines(tmp$AvgTED, tmp$reg.nl, col="red", lwd=3, xaxt='n', yaxt='n')

# JUNE 9th
log.cox.snell <- log(df09$TED) - fit09$linear.predictors / fit09$scale
coef <- fit09$coefficients
fi <- coef[names(coef) == "df09$AvgTED"] + coef[names(coef) == "(Intercept)"] + 
  fit09$scale * log.cox.snell

plot(df09$AvgTED, fi, main="June 9th", xaxt='n', yaxt='n', xlab='', ylab='')
mtext(side = 1, text = "Avg. last five TEDs", line = 0.7)
mtext(side = 2, text = "f(AvgTED)", line = 0.7)

reg.nonlinear <- lm(fi ~ I((0.1+df09$AvgTED)^(-1)) + df09$AvgTED)
reg.linear <- lm(fi ~ df09$AvgTED)
tmp <- data.frame(df09$AvgTED, fi, reg.nonlinear$fitted.values, reg.linear$fitted.values)
names(tmp) <- c("AvgTED", "fi", "reg.nl", "reg.lin")
tmp <- tmp[order(tmp$AvgTED), ]
lines(tmp$AvgTED, tmp$reg.lin, col="blue", lwd=3, xaxt='n', yaxt='n')
lines(tmp$AvgTED, tmp$reg.nl, col="red", lwd=3, xaxt='n', yaxt='n')

dev.off()

### Model improvement
# Fit model
fit06 <- survreg(tedSurv06.xlm.e3 ~ df06$OrderImb + 
                   df06$Spread + 
                   df06$XLM + 
                   df06$MO_ask + 
                   df06$MO_bid + 
                   df06$LastTED + 
                   I(0.1+df06$AvgTED^(-1))+df06$AvgTED, dist='weibull')
summary(fit06)

fit09 <- survreg(tedSurv09.xlm.e3 ~ df09$OrderImb + 
                   df09$Spread + 
                   df09$XLM + 
                   df09$MO_ask + 
                   df09$MO_bid + 
                   df09$LastTED + 
                   I(0.1+df09$AvgTED^(-1))+df09$AvgTED, dist='weibull')
summary(fit09)



### Fit Gamma GLM to survival times
par(mfrow=c(1,1))
plot(ecdf(time.06.e3.is), lwd=3)
for(i in seq(1,10,2)){
  for(j in seq(0.5,4,0.5))
  i = 3
  x.gamma <- rgamma(10000, shape=1.0/i, scale=j)
  lines(ecdf(x.gamma), col=i+1)
}



