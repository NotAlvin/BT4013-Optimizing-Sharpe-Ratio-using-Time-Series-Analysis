# import libraries
library(fGarch)
library(rugarch)
library(data.table)
library(dplyr)
library(tidyr)
library(forecast)
library(tseries)

# import data
data <- fread('./tickerData/F_ES.txt')

# take data from 2010-2020
data$DATE <- as.Date(as.character(data$DATE),"%Y%m%d")
data$year = as.numeric(format(data$DATE,'%Y'))
data = data[data$year >=2019]
data = data[data$year<=2020]
date = data$DATE
x <- data$CLOSE

plot(date, x, main='Daily Price', xlab='time', ylab='Price ($)', col='blue')
par(mar=c(1,1,1,1))
par(mfrow=c(2,1))
acf(x, main='ACF of x',lag=100) # no clear cutoff
pacf(x, main='PACF of x') # 1,2,7,8,10

log.x <- log(x)
par(mfrow=c(2,1))
acf(log.x, lag=100) # no cutoff
pacf(log.x) # 1,2,7,8,10

# check for unit root (trend)
adf.test(log.x) # p-value = 0.2932 > 0.05
                # => there is a unit root i.e. trend

# we diff the data to remove trend
diff.log.x <- diff(log.x)
plot(diff.log.x)  # seems to have no seasonality
                  # but variance spikes in certain areas
par(mfrow=c(2,1))
acf(diff.log.x)
# 1,2   4,5,6,7,8,9
# 2 or 9?
# -ve corr. at t=-1 --> MA?

pacf(diff.log.x)
# 1,2,3,4   6,7   9
# -ve corr. at t=-1
# We try p=4:9,q=2:9

# check if residuals are autocorrelated
box.test <- Box.test(diff.log.x, lag=log(length(diff.log.x)), type="Ljung")
# p-value of 2.2e-16 < 0.05
# => there is still autocorrelation in the residuals

# estimate order of GARCH model
z <- diff.log.x - mean(diff.log.x)
plot(date[2:length(date)],z)
acf(z**2, lag=70) # consecutive to 15
pacf(z**2) # 1,2    5,6 
# so we try m = 0:15, r = 2:6

# from s9_6
train.arma.garch.helper = function(p,q,m,r,data,ssolver) {
  model = NULL
  tryCatch({
    m.model = list(armaOrder =c(p,q),include.mean=T) 
    v.model = list(garchOrder=c(m,r))
    model = ugarchfit(
              ugarchspec(mean.model=m.model, 
                         variance.model=v.model),
                      data, solver=ssolver)
  }, warning = function(w) { message(w); model = NULL }, 
  error   = function(e) { message(e); model = NULL  }, 
  finally = {model})		
}

train.arma.garch = function(p,q,m,r,data) {
  solvers=c('solnp','nlminb','lbfgs','gosolnp','nloptr','hybrid')
  for (s in solvers) {
    model = train.arma.garch.helper(p,q,m,r,data,s)
    if (!is.null(model)) return(model)
  }
  return(NULL)
}


# fn to perform grid search
grid.search.arma.garch = function(start.p, end.p, start.q, end.q,
                                  start.m, end.m, start.r, end.r, data) {
  best.model = NULL 
  for (p in start.p:end.p) for (q in start.q:end.q) 
    for (m in start.m:end.m) for (r in start.r:end.r) {
      
      model = train.arma.garch(p,q,m,r,data)
      
      if (is.null(model)) {
        cat(p,q,m,r,'FAILED\n')		
      } else {
        aic = infocriteria(model)[1]
        cat(p,q,m,r,'AIC:',aic,'\n')
        if (is.null(best.model) || infocriteria(model)[1] < infocriteria(best.model)[1]) {
          best.model = model
        }
      }
    }
  best.model
}

# possible to train here, then reset the attributes of the 
# model object in Python to match the coefs obtained here?
# should I be running it on diff.log.x instead? 
# since d is not taken into account here
best.model <- grid.search.arma.garch(4,5,2,2,
                                     2,3,2,2,log.x)
# returns 4 3 2 2

# implement just ARCH(m)
library(fGarch)
arch = garchFit(~garch(2,2), data=z, trace=F)
print(arch)
ics=arch@fit$ics #aic = ics[1] , bic = ics[2]
cat('aic:', ics[1], '\n')
arch@fit$matcoef
sum(residuals(arch))**2 # 7.054765

