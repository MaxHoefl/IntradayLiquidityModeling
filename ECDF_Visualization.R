require(graphics)
rm(list=ls())
dev.off()
op <- par(mfrow = c(3, 1), mgp = c(1.5, 0.8, 0), mar =  .1+c(3,3,2,1))

F10 <- ecdf(rnorm(10))
F11 <- ecdf(rgamma(15, shape=0.5))
#summary(F10)

#plot(F10)
par(mfrow=c(2,1))
plot(F10, verticals = TRUE, do.points = FALSE)
plot(F11, verticals = TRUE, do.points = FALSE)
F10(0.0)
F11(0.0)


setwd("/Users/mh/Documents/CSML/Masterarbeit/tmp/")
rm(list=ls())
dev.off()
par(mfrow=c(1,1))

files <- list.files(path="/Users/mh/Documents/CSML/Masterarbeit/tmp", pattern="*", full.names=T, recursive=FALSE)

x1 <- seq(0.00001,0.02, 0.0001)
x2 <- seq(0.02,5,0.01)
x <- c(x1,x2)
df <- data.frame(x)


for(file in files){
  times <- read.csv(file)
  times <- times$IntertradeTimes
  times <- times[times > 0]  
  f <- ecdf(times)
  y <- f(x)
  df[substr(file, 46,55)] <- y
  #lines(x,y, log='x', type='l')
}

bool <- !grepl("2014-06-09", names(df))
bool[1] <- FALSE

df["mean"] <- rowMeans(df[,bool], na.rm=FALSE, dims=1)

bool <- !grepl("2014-06-09", names(df))
bool[1] <- FALSE
bool[length(bool)] <- FALSE


df["stddev"] <- apply(df[,bool], 1, sd)
df["lower.conf"] <- df$mean - 1.96 * df$stddev
df["upper.conf"] <- df$mean + 1.96 * df$stddev

jpeg("/Users/mh/Documents/CSML/Masterarbeit/tex/img/IntertradeTimes_highres.jpg", width = 7, height =5, units = 'in', res = 500)

plot(log(x), df$mean, lwd=3, type = 'l', ylab = "ecdf of intertrade times", 
          xlab=expression(log(t[i]-t[i-1])))
lines(log(x), df$upper.conf, lwd=3, lty='dashed')
lines(log(x), df$lower.conf, lwd=3, lty='dashed')
lines(log(x), df$`2014-06-09`, lwd=3, col='red')
legend(-12, 0.8,
       c("Mean", "95% CIs", "June 9th"), lwd=c(3,3,3,3),
       col=c("black", "black", "red"),
       lty=c("solid","dashed","solid"),
       #lty=c("solid", "dashed", "dashed", "solid"),
       bty="n",
       cex=1)

dev.off()

#### Scatter plot for activations of LSTM
# Load activations
df <- read.csv("activations")
head(df)



