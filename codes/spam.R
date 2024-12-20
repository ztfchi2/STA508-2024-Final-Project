
#install.packages("scales")

library(readr)
library(scales)
library(tidyverse)
library(rsample)
library(randomForest)

library(caret)
library(xrf)
library(parsnip)
library(rules)

library(nodeHarvest)
library(inTrees)
library(sirus)
library(e2tree)
library(ranktreeEnsemble)


############################## Read Data ##############################

setwd("C:/study/data/tree")
dat <- read_csv("spam.csv")


dat <- dat[,c(1:48, 58)]
dat$y <- factor(dat$class)
dat$class <- NULL
dat <- as.data.frame(dat)


############################## Train-test split ##############################

set.seed(123)
dat.split <- dat %>% initial_split(prop=0.70)
dat.train <- dat.split %>% training()
dat.test <- dat.split %>% testing()

x.dat.train <- dat.train %>% select(-y)
x.dat.test <- dat.test %>% select(-y)

target.dat.train <- factor(dat.train$y)
target.dat.test <- factor(dat.test$y)


############################## Results ##############################

## random forest (baseline, not rule-extraction method)
set.seed(123)
start.time <- Sys.time()
dat.y <- randomForest(factor(y) ~ ., data=dat.train, importance=TRUE)
end.time <- Sys.time()
total.time <- end.time - start.time

dat.test.pred <- stats::predict(dat.y, newdata=x.dat.test)

total.time  # time for fitting: 12.07522 secs
confusionMatrix(dat.y$predicted, reference=target.dat.train)  # Train Accuracy
confusionMatrix(dat.test.pred, reference=target.dat.test)     # Test Accuracy
# Accuracy : 0.9377
# Sensitivity : 0.9614
# Specificity : 0.9020
# Precision : 0.9366

# Balanced Accuracy : 0.9317
# G-mean : sqrt(0.9614458*0.9019964) = 0.9312
# F1-score : 0.9489
# Kappa : 0.8693



## RuleFit
set.seed(123)
start.time <- Sys.time()
dat.y.rule <- rule_fit(mode="classification") %>% fit(factor(y) ~ ., data=dat.train)
#dat.y.rule <- gbm.fit(x=x.dat.train, y=target.dat.train) %>% rulefit(n.trees=500)
end.time <- Sys.time()
total.time <- end.time - start.time

start.time.tr <- Sys.time()
dat.train.rule.pred <- stats::predict(dat.y.rule, new_data=x.dat.train)
end.time.tr <- Sys.time()
total.time.tr <- end.time.tr - start.time.tr

start.time.ts <- Sys.time()
dat.test.rule.pred <- stats::predict(dat.y.rule, new_data=x.dat.test)
end.time.ts <- Sys.time()
total.time.ts <- end.time.ts - start.time.ts

total.time    # time for fitting: 14.37848 secs
total.time.tr # time for prediction (train): 3.229096 secs
total.time.ts # time for prediction (test): 1.378333 secs

confusionMatrix(dat.train.rule.pred$.pred_class, reference=target.dat.train)  # Train Accuracy
confusionMatrix(dat.test.rule.pred$.pred_class, reference=target.dat.test)   # Test Accuracy
dat.y.rule %>% tidy(penalty=0.1) # readable rules: (335 rules)
# Accuracy : 0.8668
# Sensitivity : 0.8855
# Specificity : 0.8385
# Precision : 0.8920

# Balanced Accuracy : 0.8620
# G-mean : sqrt(0.8855422*0.8384755) = 0.8617
# F1-score : 0.8888
# Kappa : 0.7227



## NodeHarvest
set.seed(123)
start.time <- Sys.time()
dat.y.harv <- nodeHarvest(x.dat.train, as.numeric(target.dat.train)-1)   # target should be numeric?
end.time <- Sys.time()
total.time <- end.time - start.time

dat.train.harv.pred <- factor(round(dat.y.harv$predicted))
dat.test.harv.pred <- stats::predict(dat.y.harv, newdata=x.dat.test) %>% round() %>% factor()

total.time # time for fitting: 20.11045 secs
confusionMatrix(dat.train.harv.pred, reference=target.dat.train)  # Train Accuracy
confusionMatrix(dat.test.harv.pred, reference=target.dat.test)    # Test Accuracy

plot(dat.y.harv) # readable rules (47 rules)
print(dat.y.harv, nonodes=10)
# Accuracy : 0.8733
# Sensitivity : 0.9614
# Specificity : 0.7405
# Precision : 0.8480

# Balanced Accuracy : 0.8510
# G-mean : sqrt(0.9614458*0.7404719) = 0.8438
# F1-score : 0.9012
# Kappa : 0.7265



## inTrees
set.seed(123)
start.time <- Sys.time()  # only time for rule extraction, random forest model used
intree.lists <- dat.y %>% RF2List()
end.time <- Sys.time()
total.time <- end.time - start.time

start.time.rules <- Sys.time()
intree.rules <- extractRules(intree.lists, x.dat.train)
intree.rule.metric <- getRuleMetric(intree.rules, x.dat.train, target.dat.train)
intree.rule.metric <- pruneRule(intree.rule.metric, x.dat.train, target.dat.train)
end.time.rules <- Sys.time()
total.time.rules <- end.time.rules - start.time.rules

start.time.learn <- Sys.time()
dat.y.intree <- buildLearner(intree.rule.metric, x.dat.train, target.dat.train)
end.time.learn <- Sys.time()
total.time.learn <- end.time.learn - start.time.learn

dat.train.intree.pred <- applyLearner(dat.y.intree, x.dat.train) %>% factor()
dat.test.intree.pred <- applyLearner(dat.y.intree, x.dat.test) %>% factor()

total.time + total.time.learn # time for model fit: 10.62985 secs
total.time.rules  # time for rule extraction: 8.579971 secs
confusionMatrix(dat.train.intree.pred, reference=target.dat.train)  # Train Accuracy
confusionMatrix(dat.test.intree.pred, reference=target.dat.test)    # Test Accuracy

presentRules(dat.y.intree, colnames(x.dat.train))  # readable rules (43 rules)
# Accuracy : 0.9138
# Sensitivity : 0.9506
# Specificity : 0.8584
# Precision : 0.9100

# Balanced Accuracy : 0.9045
# G-mean : sqrt(0.9506024*0.8584392) = 0.9033
# F1-score : 0.9299
# Kappa : 0.8183



## SIRUS
set.seed(123)
start.time <- Sys.time()
dat.y.sirus <- sirus.fit(x.dat.train, as.numeric(target.dat.train)-1, # designed for <= 100 rules
                         type="classif", num.rule=10)  # need numeric targets... only takes binary
end.time <- Sys.time()
total.time <- end.time - start.time

dat.train.sirus.pred <- sirus.predict(dat.y.sirus, x.dat.train) %>% round() %>% factor()
dat.test.sirus.pred <- sirus.predict(dat.y.sirus, x.dat.test) %>% round() %>% factor()

total.time  # time for fitting: 1.551524 secs
confusionMatrix(dat.train.sirus.pred, reference=target.dat.train) # Train Accuracy
confusionMatrix(dat.test.sirus.pred, reference=target.dat.test) # Test Accuracy

sirus.print(dat.y.sirus) # readable rules (10 rules)
# Accuracy : 0.8219
# Sensitivity : 0.9554
# Specificity : 0.6207
# Precision : 0.7914

# Balanced Accuracy : 0.7881
# G-mean : sqrt(0.9554217*0.6206897) = 0.7701
# F1-score : 0.8657
# Kappa : 0.608



## E2Tree
set.seed(123)
start.time <- Sys.time()
dat.ensemble <- randomForest(factor(y) ~ ., data=dat.train, importance=TRUE, proximity=TRUE)
end.time <- Sys.time()
total.time <- end.time - start.time

start.time.dis <- Sys.time()
#D <- createDisMatrix(dat.ensemble, data=dat.train, label="y", parallel=TRUE)
end.time.dis <- Sys.time()
total.time.dis <- end.time.dis - start.time.dis
# takes too much time, not appropriate for large dataset?

start.time.et <- Sys.time()
dat.y.e2tree <- e2tree(factor(y) ~ ., data=dat.train, 
                       D=D, ensemble=dat.ensemble)
end.time.et <- Sys.time()
total.time.et <- end.time.et - start.time.et

dat.train.e2tree.pred <- ePredTree(dat.y.e2tree, x.dat.train, target="y")$fit %>% factor()
dat.test.e2tree.pred <- ePredTree(dat.y.e2tree, x.dat.test, target="y")$fit %>% factor()

total.time      # time for fitting: 20.47957 secs
total.time.et   # time for predict: 
total.time.dis  # time for disdatilarity matrix calculation: 
confusionMatrix(dat.train.e2tree.pred, reference=target.dat.train) # Train Accuracy
confusionMatrix(dat.test.e2tree.pred, reference=target.dat.test) # Test Accuracy

rpart2Tree(dat.y.e2tree, dat.ensemble) %>% rpart.plot::rpart.plot(tweak=1.7) # tree plot, 65 no 32 rules
# No analysis here



## RankTree
set.seed(123)
start.time <- Sys.time()
dat.rank.ensemble <- rforest(y ~ ., data=dat.train, importance=TRUE, dimreduce=FALSE)
# dimension reduction option: rule extraction speed up, accuracy down?
end.time <- Sys.time()
total.time <- end.time - start.time

start.time.ext <- Sys.time()
dat.y.rank <- extract.rules(dat.rank.ensemble)
end.time.ext <- Sys.time()
total.time.ext <- end.time.ext - start.time.ext

dat.train.rank.pred <- ranktreeEnsemble::predict(dat.rank.ensemble, newdata=x.dat.train)
dat.test.rank.pred <- ranktreeEnsemble::predict(dat.rank.ensemble, newdata=x.dat.test)


total.time      # time for model fit: 5.132068 secs
total.time.ext  # time for rule extraction: 6.469567 mins
confusionMatrix(factor(dat.train.rank.pred$label), target.dat.train)  # Train Accuracy
confusionMatrix(factor(dat.test.rank.pred$label), target.dat.test)  # Test Accuracy

dat.y.rank$rule  # readable rules (20 rules)
# Accuracy : 0.9377
# Sensitivity : 0.9482
# Specificity : 0.9220
# Precision : 0.9482

# Balanced Accuracy : 0.9351
# G-mean : sqrt(0.9481928*0.9219601) = 0.9350
# F1-score : 0.9482
# Kappa : 0.8702


