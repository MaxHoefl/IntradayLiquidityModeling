##### NET ORDERFLOW VISUALIZATION ########

rm(list=ls())
setwd("/Users/mh/Documents/CSML/Masterarbeit/Python/")

### BEFORE STOCK SPLIT
df.before <- read.csv("NETORDERFLOW_DATA_BEFORESPLIT.csv")
df.before <- df.before[, !names(df.before) %in% c("X")]
head(df.before)

mean.before <- colMeans(df.before, na.rm=FALSE, dims=1L)
sd.before <- apply(df.before, 2, sd)
lower.conf.before <- mean.before - 1.96 * sd.before
upper.conf.before <- mean.before + 1.96 * sd.before

jpeg("/Users/mh/Documents/CSML/Masterarbeit/tex/img/NetOrderFlow_highres.jpg", width = 7, height =5, units = 'in', res = 500)
plot(mean.before, lwd = 3, type = 'l', ylim=c(-200,200), xaxt='n', xlab=expression(tau),
     ylab="Mean Net Order Flow")
axis(side=1, at=c(1:7), labels=c('1e-6','1e-5','1e-4','1e-3','1e-2','1e-1','1e-0'))
par(new=TRUE)
plot(lower.conf.before, lwd = 3, lty='dashed', type='l', yaxt='n', xaxt='n', ylab='', xlab='',
     ylim=c(-1000, 1000))
lines(upper.conf.before, lwd = 3, lty='dashed', type='l', yaxt='n', xaxt='n', ylab='', xlab='')
axis(4)
mtext("CI of Net Order Flow", side=4, line=3)

### AFTER STOCK SPLIT
df.after <- read.csv("NETORDERFLOW_DATA_AFTERSPLIT.csv")
df.after <- df.after[, !names(df.after) %in% c("X")]
head(df.after)

mean.after <- colMeans(df.after, na.rm=FALSE, dims=1L)
sd.after <- apply(df.after, 2, sd)
lower.conf.after <- mean.after - 1.96 * sd.after
upper.conf.after <- mean.after + 1.96 * sd.after

lines(mean.after, lwd = 3, type = 'l', ylim=c(-200,200), xaxt='n', xlab=expression(tau), col='red')
#axis(side=1, at=c(1:7), labels=c('1e-6','1e-5','1e-4','1e-3','1e-2','1e-1','1e-0'))
par(new=TRUE)
plot(lower.conf.after, lwd = 3, lty='dashed', type='l', yaxt='n', xaxt='n', ylab='', xlab='',
     ylim=c(-1000, 1000), col='red')
lines(upper.conf.after, lwd = 3, lty='dashed', type='l', yaxt='n', xaxt='n', ylab='', xlab='',
      col='red')

legend("bottomleft",
       c("before", "95% CIs", "after", "95% CIs"), lwd=c(3,3,3,3,3),
       col=c("black", "black", "red", "red"),
       lty=c("solid","dashed","solid", "dashed"),
       #lty=c("solid", "dashed", "dashed", "solid"),
       x.intersp=0.5,
       y.intersp=0.8,
       cex=1)

dev.off()

