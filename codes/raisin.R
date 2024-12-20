
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
dat <- read_csv("Raisin_Dataset.csv")


dat$y <- case_when(dat$Class == "Kecimen" ~ 0, 
                     dat$Class == "Besni" ~ 1) # unequal class y


dat$y <- factor(dat$y)
dat <- dat[,-8]
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

total.time  # time for fitting: 0.4075768 secs
confusionMatrix(dat.y$predicted, reference=target.dat.train)  # Train Accuracy
confusionMatrix(dat.test.pred, reference=target.dat.test)     # Test Accuracy



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

total.time    # time for fitting: 7.500478 secs
total.time.tr # time for prediction (train): 0.8155119 secs
total.time.ts # time for prediction (test): 0.751322 secs

confusionMatrix(dat.train.rule.pred$.pred_class, reference=target.dat.train)  # Train Accuracy
confusionMatrix(dat.test.rule.pred$.pred_class, reference=target.dat.test)   # Test Accuracy

dat.y.rule %>% tidy(penalty=0.1) # readable rules: (255 rules)



## NodeHarvest
set.seed(123)
start.time <- Sys.time()
dat.y.harv <- nodeHarvest(x.dat.train, as.numeric(target.dat.train)-1)   # target should be numeric?
end.time <- Sys.time()
total.time <- end.time - start.time

dat.train.harv.pred <- factor(round(dat.y.harv$predicted))
dat.test.harv.pred <- stats::predict(dat.y.harv, newdata=x.dat.test) %>% round() %>% factor()

total.time # time for fitting: 2.866532 secs
confusionMatrix(dat.train.harv.pred, reference=target.dat.train)  # Train Accuracy
confusionMatrix(dat.test.harv.pred, reference=target.dat.test)    # Test Accuracy

plot(dat.y.harv) # readable rules (51 rules)
print(dat.y.harv, nonodes=10)



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


total.time + total.time.learn # time for model fit: 7.714107 secs
total.time.rules  # time for rule extraction: 5.301799 secs
confusionMatrix(dat.train.intree.pred, reference=target.dat.train)  # Train Accuracy
confusionMatrix(dat.test.intree.pred, reference=target.dat.test)    # Test Accuracy

presentRules(dat.y.intree, colnames(x.dat.train))  # readable rules (38 rules)



## SIRUS
set.seed(123)
start.time <- Sys.time()
dat.y.sirus <- sirus.fit(x.dat.train, as.numeric(target.dat.train)-1, # designed for <= 100 rules
                         type="classif", num.rule=10)  # need numeric targets... only takes binary
end.time <- Sys.time()
total.time <- end.time - start.time

dat.train.sirus.pred <- sirus.predict(dat.y.sirus, x.dat.train) %>% round() %>% factor()
dat.test.sirus.pred <- sirus.predict(dat.y.sirus, x.dat.test) %>% round() %>% factor()

total.time  # time for fitting: 0.482378 secs
confusionMatrix(dat.train.sirus.pred, reference=target.dat.train) # Train Accuracy
confusionMatrix(dat.test.sirus.pred, reference=target.dat.test) # Test Accuracy

sirus.print(dat.y.sirus) # readable rules (10 rules)



## E2Tree
set.seed(123)
start.time <- Sys.time()
dat.ensemble <- randomForest(factor(y) ~ ., data=dat.train, importance=TRUE, proximity=TRUE)
end.time <- Sys.time()
total.time <- end.time - start.time

start.time.dis <- Sys.time()
D <- createDisMatrix(dat.ensemble, data=dat.train, label="y", parallel=TRUE)
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

total.time      # time for fitting: 1.33368 secs
total.time.et   # time for predict: 53.66614 secs
total.time.dis  # time for disdatilarity matrix calculation: 2.215602 mins
confusionMatrix(dat.train.e2tree.pred, reference=target.dat.train) # Train Accuracy
confusionMatrix(dat.test.e2tree.pred, reference=target.dat.test) # Test Accuracy

rpart2Tree(dat.y.e2tree, dat.ensemble) %>% rpart.plot::rpart.plot(tweak=1.7) # tree plot, 65 no 32 rules



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


total.time      # time for model fit: 
total.time.ext  # time for rule extraction:
confusionMatrix(factor(dat.train.rank.pred$label), target.dat.train)  # Train Accuracy
confusionMatrix(factor(dat.test.rank.pred$label), target.dat.test)  # Test Accuracy

dat.y.rank$rule  # readable rules (20 rules)



