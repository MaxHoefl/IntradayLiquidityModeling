######################################################
# Leaky ReLU visualization
######################################################
jpeg("/Users/mh/Documents/CSML/Masterarbeit/tex/img/LeakyReLu_highres.jpg", 
     width = 7, height =5, units = 'in', res = 500)
par(mfrow=c(2,1))
curve(0.2*x, -5, 0, axes=FALSE, ylim=c(-4,4), xlim=c(-4,4), xlab='', ylab='', lwd=3)
par(new=TRUE)
curve(1.0*x, 0, 5, axes=FALSE, ylim=c(-4,4), xlim=c(-4,4), xlab='', ylab='', lwd=3)
axis(1, pos=0, at=-5:5, col.axis = "white", labels = FALSE)
axis(2, pos=0, at=-4:4, col.axis = "white", labels = FALSE)

curve(0.2*x, -5, 0, axes=FALSE, ylim=c(-4,4), xlim=c(-4,4), xlab='', ylab='', lwd=3)
par(new=TRUE)
curve(1.0*x, 0, 5, axes=FALSE, ylim=c(-4,4), xlim=c(-4,4), xlab='', ylab='', lwd=3)
axis(1, pos=0, at=-5:5, col.axis = "white", labels = FALSE)
axis(2, pos=0, at=-4:4, col.axis = "white", labels = FALSE)

dev.off()