
#install.packages("scales")

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


############################## Data Generation ##############################

N <- 500
set.seed(111)
x1 <- rnorm(N, mean=0, sd=100)
set.seed(222)
x2 <- rnorm(N, mean=30, sd=30)

x <- 10 + 0.3*-x1 + 0.7*x2 + rnorm(N, mean=0, sd=20)

set.seed(333)
x3 <- rnorm(N, mean=-40, sd=100) + 0.4*x + 0.3*x1

set.seed(444)
x4 <- rnorm(N, mean=20, sd=20) + x2^2/60


x <- rescale(x, to=c(0,1))
y <- round(x)

sim <- data.frame(x1=round(x1, 3), x2=round(x2, 3), 
                  x3=round(x3, 3), x4=round(x4, 3), y=y)

table(sim$y)


############################## Train-test split ##############################

set.seed(1234)
sim.split <- sim %>% initial_split(prop=0.70)
sim.train <- sim.split %>% training()
sim.test <- sim.split %>% testing()

x.sim.train <- sim.train %>% select(-y)
x.sim.test <- sim.test %>% select(-y)

target.sim.train <- factor(sim.train$y)
target.sim.test <- factor(sim.test$y)


############################## Results ##############################

## random forest (baseline, not rule-extraction method)
set.seed(123)
start.time <- Sys.time()
sim.y <- randomForest(factor(y) ~ ., data=sim.train, importance=TRUE)
end.time <- Sys.time()
total.time <- end.time - start.time

sim.test.pred <- stats::predict(sim.y, newdata=x.sim.test)

total.time  # time for fitting: 0.1914032 secs
confusionMatrix(sim.y$predicted, reference=target.sim.train)  # Train Accuracy: 
confusionMatrix(sim.test.pred, reference=target.sim.test)     # Test Accuracy: 



## RuleFit
set.seed(123)
start.time <- Sys.time()
sim.y.rule <- rule_fit(mode="classification") %>% fit(factor(y) ~ ., data=sim.train)
#sim.y.rule <- gbm.fit(x=x.sim.train, y=target.sim.train) %>% rulefit(n.trees=500)
end.time <- Sys.time()
total.time <- end.time - start.time

start.time.tr <- Sys.time()
sim.train.rule.pred <- stats::predict(sim.y.rule, new_data=x.sim.train)
end.time.tr <- Sys.time()
total.time.tr <- end.time.tr - start.time.tr

start.time.ts <- Sys.time()
sim.test.rule.pred <- stats::predict(sim.y.rule, new_data=x.sim.test)
end.time.ts <- Sys.time()
total.time.ts <- end.time.ts - start.time.ts

total.time    # time for fitting: 6.265336 secs
total.time.tr # time for prediction (train): 0.8555441 secs
total.time.ts # time for prediction (test): 0.7884262 secs

confusionMatrix(sim.train.rule.pred$.pred_class, reference=target.sim.train)  # Train Accuracy: 
confusionMatrix(sim.test.rule.pred$.pred_class, reference=target.sim.test)   # Test Accuracy: 

sim.y.rule %>% tidy(penalty=0.1) # readable rules: (232 rules)



## NodeHarvest
set.seed(123)
start.time <- Sys.time()
sim.y.harv <- nodeHarvest(x.sim.train, as.numeric(target.sim.train)-1)   # target should be numeric?
end.time <- Sys.time()
total.time <- end.time - start.time

sim.train.harv.pred <- factor(round(sim.y.harv$predicted))
sim.test.harv.pred <- stats::predict(sim.y.harv, newdata=x.sim.test) %>% round() %>% factor()

total.time # time for fitting: 4.405447 secs
confusionMatrix(sim.train.harv.pred, reference=target.sim.train)  # Train Accuracy: 
confusionMatrix(sim.test.harv.pred, reference=target.sim.test)    # Test Accuracy: 

plot(sim.y.harv) # readable rules (31 rules)
print(sim.y.harv, nonodes=10)



## inTrees
set.seed(123)
start.time <- Sys.time()  # only time for rule extraction, random forest model used
intree.lists <- sim.y %>% RF2List()
end.time <- Sys.time()
total.time <- end.time - start.time

start.time.rules <- Sys.time()
intree.rules <- extractRules(intree.lists, x.sim.train)
intree.rule.metric <- getRuleMetric(intree.rules, x.sim.train, target.sim.train)
intree.rule.metric <- pruneRule(intree.rule.metric, x.sim.train, target.sim.train)
end.time.rules <- Sys.time()
total.time.rules <- end.time.rules - start.time.rules

start.time.learn <- Sys.time()
sim.y.intree <- buildLearner(intree.rule.metric, x.sim.train, target.sim.train)
end.time.learn <- Sys.time()
total.time.learn <- end.time.learn - start.time.learn


sim.train.intree.pred <- applyLearner(sim.y.intree, x.sim.train) %>% factor()
sim.test.intree.pred <- applyLearner(sim.y.intree, x.sim.test) %>% factor()


total.time + total.time.learn       # time for model fit: 4.745535 secs
total.time.rules  # time for rule extraction: 4.898217 secs
confusionMatrix(sim.train.intree.pred, reference=target.sim.train)  # Train Accuracy: 
confusionMatrix(sim.test.intree.pred, reference=target.sim.test)    # Test Accuracy: 

presentRules(sim.y.intree, colnames(x.sim.train))  # readable rules (29 rules)



## SIRUS
set.seed(123)
start.time <- Sys.time()
sim.y.sirus <- sirus.fit(x.sim.train, as.numeric(target.sim.train)-1, # designed for <= 100 rules
                              type="classif", num.rule=10)  # need numeric targets... only takes binary
end.time <- Sys.time()
total.time <- end.time - start.time

sim.train.sirus.pred <- sirus.predict(sim.y.sirus, x.sim.train) %>% round() %>% factor()
sim.test.sirus.pred <- sirus.predict(sim.y.sirus, x.sim.test) %>% round() %>% factor()

total.time  # time for fitting: 0.7647889 secs
confusionMatrix(sim.train.sirus.pred, reference=target.sim.train) # Train Accuracy:  
confusionMatrix(sim.test.sirus.pred, reference=target.sim.test) # Train Accuracy: 

sirus.print(sim.y.sirus) # readable rules (10 rules)

 

## E2Tree
set.seed(123)
start.time <- Sys.time()
sim.ensemble <- randomForest(factor(y) ~ ., data=sim.train, importance=TRUE, proximity=TRUE)
end.time <- Sys.time()
total.time <- end.time - start.time

start.time.dis <- Sys.time()
D <- createDisMatrix(sim.ensemble, data=sim.train, label="y", parallel=TRUE)
end.time.dis <- Sys.time()
total.time.dis <- end.time.dis - start.time.dis
# takes too much time, not appropriate for large dataset?

start.time.et <- Sys.time()
sim.y.e2tree <- e2tree(factor(y) ~ ., data=sim.train, 
                            D=D, ensemble=sim.ensemble)
end.time.et <- Sys.time()
total.time.et <- end.time.et - start.time.et

sim.train.e2tree.pred <- ePredTree(sim.y.e2tree, x.sim.train, target="y")$fit %>% factor()
sim.test.e2tree.pred <- ePredTree(sim.y.e2tree, x.sim.test, target="y")$fit %>% factor()

total.time      # time for fitting: 0.206316 secs
total.time.et   # time for predict: 
total.time.dis  # time for dissimilarity matrix calculation: 1.265243 mins = 75.91458 secs
confusionMatrix(sim.train.e2tree.pred, reference=target.sim.train) # Train Accuracy:  
confusionMatrix(sim.test.e2tree.pred, reference=target.sim.test) # Test Accuracy: 

rpart2Tree(sim.y.e2tree, sim.ensemble) %>% rpart.plot::rpart.plot(tweak=1.5) # tree plot, 65 no 32 rules



## RankTree
set.seed(123)
start.time <- Sys.time()
sim.rank.ensemble <- rforest(y ~ ., data=sim.train, importance=TRUE, dimreduce=FALSE)
# dimension reduction option: rule extraction speed up, accuracy down?
end.time <- Sys.time()
total.time <- end.time - start.time

start.time.ext <- Sys.time()
sim.y.rank <- extract.rules(sim.rank.ensemble)
end.time.ext <- Sys.time()
total.time.ext <- end.time.ext - start.time.ext

sim.train.rank.pred <- ranktreeEnsemble::predict(sim.rank.ensemble, newdata=x.sim.train)
sim.test.rank.pred <- ranktreeEnsemble::predict(sim.rank.ensemble, newdata=x.sim.test)


total.time      # time for model fit: 3.671507 secs
total.time.ext  # time for rule extraction:
confusionMatrix(factor(sim.train.rank.pred$label), target.sim.train)  # Train Accuracy : 0.9794 
confusionMatrix(factor(sim.test.rank.pred$label), target.sim.test)  # Test Accuracy : 0.9348

sim.y.rank$rule  # readable rules (20 rules)



