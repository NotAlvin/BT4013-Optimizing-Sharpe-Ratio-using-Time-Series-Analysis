library(forecast)

filename <- 'F_ES.txt'

library(data.table)
library(dplyr)
library(tidyr)
# library(rstudioapi)

setwd("~/Desktop/NUS/Y2S2/BT4013/Project/Project Documents/bt4013_proj")
setwd("./tickerData")

data <- fread(filename)

data$DATE <- as.Date(as.character(data$DATE),"%Y%m%d")
data$year = as.numeric(format(data$DATE,'%Y'))
data_2010_to_2020 = data[data$year >=2020]
data_2010_to_2020 = data_2010_to_2020[data_2010_to_2020$year<=2021]

x <- data_2010_to_2020$CLOSE

plot(x)

log.x <- log(x)
plot(log.x)

par(mfrow=c(2,1))
acf(log.x) #cut off at 9
pacf(log.x) #cut off at 9

#Let's test for unit root (to be sure there's a trend)
library(tseries)
for (p in 1:10) {
  adf.out = adf.test(log.x, k=p)
  print(paste("###### ADF TEST for p =", p, ' ; p-value: ', adf.out$p.value, adf.out$p.value<0.05))
}

diff.log.x <- diff(log.x)
plot(diff.log.x)

par(mfrow=c(2,1))
acf(diff.log.x) #cut off at 9
pacf(diff.log.x) #cut off at 9

min.aic.info = list(p=-1, q=-1, d=-1,
                    sse=.Machine$double.xmax, res.p.value=100, model=NULL)

for (d in 1:1) for (p in 0:9) for (q in 0:9) {
  
  if (p+q+d >= 8) next #rule of thumb; also determine by computational resources
  
  model = tryCatch({
    arima(log.x,  order=c(p,d,q), include.mean = FALSE)
  }, warning = function(w) { message(w) }, 
  error   = function(e) { message(e); return(NULL) }, 
  finally = {})
  
  if (!is.null(model)) {
    box.test = Box.test(model$residuals, lag=log(length(model$residuals)))
    print(paste('####### p,q:', p, q, "#########"))
    print(paste('aic:  ', model$aic))
    print(paste('logl: ', model$loglik))				
    print(paste('Portmanteau test on residuals',box.test$p.value))					
    #print(model$coef)		
    
    if (is.null(min.aic.info$model) || 
        model$aic < (min.aic.info$model)$aic ||
        (model$aic == (min.aic.info$model)$aic && 
         p+q+d < min.aic.info$p + min.aic.info$d + min.aic.info$q
         )){
      min.aic.info$p = p
      min.aic.info$q = q
      min.aic.info$d = d
      min.aic.info$res.p.value = box.test$p.value			
      min.aic.info$model = model								
    }		
  }
}

#Pick the model with the lowest AIC
print("#################")
min.aic.model = min.aic.info$model
print(paste('min.aic.p  : ', min.aic.info$p))
print(paste('min.aic.q  : ', min.aic.info$q))
print(paste('min.aic.d  : ', min.aic.info$d))
print(paste('min.aic    : ', min.aic.model$aic))
print(paste('min.aic.res.p.value: ', min.aic.info$res.p.value))
print('min.aic.coefficients')
print(min.aic.model$coef)
print(paste('sigma2: ', min.aic.model$sigma2))

