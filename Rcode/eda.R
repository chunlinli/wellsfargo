## ======= Update at July 25 ========== Yu ##

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
train <- read.csv("../data/train.csv")
test <- read.csv("../data/test.csv")
library(dplyr)

## make naive predictions on the train set
train <- within(train, {
  XC <- as.numeric(XC) - 3
  Xg <- X2 + X3 + X4 + X6 + X7 + X11 + X15 + X17 + X19 + X20 + 
    X21 + X22 + X25 + X26 + X27 + XC + 2
})

plot(y ~ Xg, train)
abline(v=0)

# check error rate
train.pred <- as.numeric(train$Xg < 0)
(err.rate <- mean(train.pred != train$y)) 

# check 1 vs 0 ratio in train$y
mean(train$y)

## check the margin of the classifier
# upper part
(upper <- train %>% 
  filter(Xg > 0) %>% 
  select(Xg) %>% 
  summarise(upper.margin=min(Xg), upper.count=n()))
# lower part
(lower <- train %>% 
  filter(Xg < 0) %>% 
  select(Xg) %>% 
  summarise(lower.margin=max(Xg), lower.count=n()))

# make naive predictions on the test set
test <- within(test, {
  XC <- as.numeric(XC) - 3
  Xg <- X2 + X3 + X4 + X6 + X7 + X11 + X15 + X17 + X19 + X20 + 
    X21 + X22 + X25 + X26 + X27 + XC + 2
})

test.pred <- as.numeric(test$Xg < 0)

# check 1 vs 0 raio in test.pred
mean(test.pred)

# check how many Xg in test would fall in the margin
test %>% 
  select(Xg) %>%
  filter(Xg >= lower$lower.margin, Xg <= upper$upper.margin)
  

## ======= Update at July 20 ========== Yu ##
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
train <- read.csv("../data/train.csv")
test <- read.csv("../data/test.csv")

# compare the density curves between train and test
par(mfrow = c(2, 3))
for (i in 1:30) {
  plot(density(train[, i]), col = 'blue', lwd=2, 
       main = paste("density curve for ", names(train)[i]))
  lines(density(test[, i]), col = 'red', lwd=2)
  legend("topright", legend = c("train", "test"), 
         col = c("blue", "red"), lwd = c(2, 2), cex=0.7)
}
par(mfrow = c(1, 1))

# do Kolmogorov-Smirnov test between train and test
for (i in 1:30) {
  result <- ks.test(train[, i], test[, i])
  if (result$p.value < 0.1) {
    print(paste("i = ", i, sep = ""))
    print(result)
  }
}

# do Mann-Whitney U test between train and test
for (i in 1:30) {
  result <- wilcox.test(train[, i], test[, i])
  if (result$p.value < 0.1) {
    print(paste("i = ", i, sep = ""))
    print(result)
  }
}


## ============================================= ##

dat <- read.csv("../data/train.csv")

### boxplot x~y
## some are significant
for (i in 1:30) {
  boxplot(dat[,i] ~ dat[,32])
}

### imbalance label
table(dat[,32])

############## explore categorical variable
dat2 = read.csv("../data/test.csv")
table(c(dat[,31],dat2[,31]))
max(table(c(dat[,31],dat2[,31]))) - min(table(c(dat[,31],dat2[,31])))
## it seems like a uniform distribution on {A,B,C,D,E}
## simulation
{
  see = sample(1:5, 10000, T)
  table(see)
  max(table(see)) - min(table(see))
}
#### plot categorical variable with y
cat0 = dat[dat$y == 0,]
cat1 = dat[dat$y == 1,]
table(cat0[,31])
table(cat1[,31])
barplot(table(cat0[,31]))
barplot(table(cat1[,31]))
## more CDE in class 0, more ABC in class1
## think about how they generated abels using this categorical variable

############# explore numeric variables
### looks like normal and no correlations
library(nortest)
### class 0
for (i in 1:30) {
  qqnorm(cat0[,i])
  qqline(cat0[,i])
  print(ad.test(cat0[,i]))
}
### class 1
for (i in 1:30) {
  qqnorm(cat1[,i])
  qqline(cat1[,i])
  print(ad.test(cat1[,i]))
}
### all training
for (i in 1:30) {
  qqnorm(dat[,i])
  qqline(dat[,i])
  print(ad.test(dat[,i]))
}

## compare with true normal
for (i in 1:30) {
  x = rnorm(3000)
  print(ad.test(x))
  qqnorm(x)
  qqline(x)
}

### cor
cor(dat[,1:30])

################# fit some model
set.seed(1)
### one hot encoder
newdat = dat[,1:30]
newdat[,31:35] = diag(rep(1,5))[as.numeric(dat[,31]),]
newdat[,36] = dat[,32]

########## logistic + lasso
library(glmnet)
mod = cv.glmnet(as.matrix(newdat[,1:35]), newdat[,36], family = "binomial",
                type.measure="class")
mod$lambda
mod$cvm #### best error rate is 0.005000, smaller lambda gives better result

mod2 = cv.glmnet(as.matrix(newdat[,1:30]), newdat[,36], family = "binomial",
                type.measure="class")
mod2$lambda
mod2$cvm #### best error rate is 0.005000, smaller lambda gives better result


## coefficients
View(as.matrix(mod$glmnet.fit$beta)[,100])
## it seems the coefficients are -2,-1,0,1,2


library(mda)
########### MDA
train_index = sample(1:3000, 2700)
train = newdat[train_index,]
test = newdat[-train_index,]

?mda
mdamod = mda(V36~., data = train[,-(31:35)])
mdamod = mda(y~., data = dat[,-(31)])
mda_pred = predict(mdamod,test[,-(31:35)])
loss = function(y, y_p){
  mean(y != y_p)
}
loss(mda_pred, test[,36]) ### error rate is about 0.1
