rm(list=ls())

# Load required libraries
library(mvtnorm)
library(plotly)

# Function to calculate log likelihood
# Parameters
n_obs <- 100
Sigma <- matrix(c(1, 0.5, 0.5, 1), nrow = 2)
X <- rmvnorm(n_obs, mean=c(4,10), sigma=Sigma) 
mu1 <- seq(-20,20, l = 100)
mu2 <- seq(-40,40, l = 100)
conds = expand.grid(mu1 = mu1, mu2= mu2)
conds$lik = NA
for(i in 1:nrow(conds)) {
  conds$lik[i] = sum(dmvnorm(X, mean=c(conds$mu1[i], conds$mu2[i]), sigma = Sigma, log=T))
}

lik_mat = tapply(conds$lik, list(conds$mu1, conds$mu2), mean)

# Create 3D contour plot
plot_ly(x = mu1, y = mu2, z = lik_mat, type = "surface",
        contours = list(z = list(show = TRUE, usecolormap = TRUE,
                                 highlightcolor = "#ff0000", project = list(z = TRUE)))) %>% 
  

which(lik_mat==max(lik_mat), arr.ind = T)



