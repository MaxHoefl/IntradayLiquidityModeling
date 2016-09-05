##########################################################################################
# Visualizations of activations in last LSTM leaky RELU layer
##########################################################################################

rm(list = ls())
setwd("/Users/mh/Documents/CSML/Masterarbeit/Python/LSTM_v2/m04/")
library(scatterplot3d)
library(rgl)
# Load activations
df.activations <- read.csv("df_activations.csv")
df.y <- read.csv("y_true_score.csv")
df.true <- df.y$y_true
df.score <- df.y$y_score

df.activations <- df.activations[, c("X", "Y", "Z")]
df.activations["true"] <- 0
df.activations[df.true==1, "true"] <- 1
df.activations$pcolor[df.activations$true==1] <- "red"
df.activations$pcolor[df.activations$true==0] <- "blue"
df.activations$ypred[df.score >= 0.4125] <- 1
df.activations$ypred[df.score < 0.4125] <- 0
df.activations$pch[df.activations$ypred == 1] <- 1
df.activations$pch[df.activations$ypred == 0] <- 2
# upper triangle filled, pch=17
# lower triangle filled, 

df_tmp <- df.activations[c(1:1000000),]

s3d <- scatterplot3d(df_tmp$X, df_tmp$Y, df_tmp$Z,        # x y and z axis
                     color=df_tmp$pcolor, pch=df_tmp$pch,        # circle color indicates no. of cylinders
                     type="h", lty.hplot=2,       # lines to the horizontal plane
                     main="Training activations",
                     xlab="X",
                     ylab="Y",
                     zlab="Z")

s3d.coords <- s3d$xyz.convert(sub_$X, sub_$Y, sub_$Z)
text(s3d.coords$x, s3d.coords$y,     # x and y coordinates
     labels=row.names(mtcars),       # text to plot
     pos=4, cex=.5)



############################################################
############################################################
# Graveyars


plot3d(df_tmp$X, df_tmp$Y, df_tmp$Z, col=df_tmp$pcolor, size=5)


df_mo <- df[df_y==1,]
df_no <- df[df_y==0,]

sub_mo <- df_mo[c(1:1000),]
sub_no <- df_no[c(1:1000),]
sub_ <- df[c(1:1000),]

par(mfrow = c(1,1))
scatterplot3d(sub_mo$X, sub_mo$Y, sub_mo$Z, type='h')
par(new=TRUE)
scatterplot3d(sub_no$X, sub_no$Y, sub_no$Z, type='h')


par(mfrow=c(1,1))
plot3d(df_mo$X, df_mo$Y, df_mo$Z, col="red", size=3)
plot3d(df_no$X, df_no$Y, df_no$Z, col="red", size=3)

par(mfrow = c(1,1))
sd3 <-scatterplot3d(sub_$X, sub_$Y, sub_$Z, 
                    type='h', lty.hplot=2)







library(scatterplot3d)

# create column indicating point color
mtcars$pcolor[mtcars$cyl==4] <- "red"
mtcars$pcolor[mtcars$cyl==6] <- "blue"
mtcars$pcolor[mtcars$cyl==8] <- "darkgreen"
with(mtcars, {
  s3d <- scatterplot3d(disp, wt, mpg,        # x y and z axis
                       color=pcolor, pch=19,        # circle color indicates no. of cylinders
                       type="h", lty.hplot=2,       # lines to the horizontal plane
                       scale.y=.75,                 # scale y axis (reduce by 25%)
                       main="3-D Scatterplot Example 4",
                       xlab="Displacement (cu. in.)",
                       ylab="Weight (lb/1000)",
                       zlab="Miles/(US) Gallon")
  s3d.coords <- s3d$xyz.convert(disp, wt, mpg)
  text(s3d.coords$x, s3d.coords$y,     # x and y coordinates
       labels=row.names(mtcars),       # text to plot
       pos=4, cex=.5)                  # shrink text 50% and place to right of points)
  # add the legend
  legend("topleft", inset=.05,      # location and inset
         bty="n", cex=.5,              # suppress legend box, shrink text 50%
         title="Number of Cylinders",
         c("4", "6", "8"), fill=c("red", "blue", "darkgreen"))
