
############# [Settings] Dependencies #############

## install packages
install.packages("readr")
install.packages("tidyverse")
install.packages("rsample")
install.packages("caret")

install.packages("randomForest")
install.packages("rules")
install.packages("xrf")
install.packages("parsnip")       # RuleFit: rule_fit()
#remotes::install_github("gravesee/rulefit")

#install.packages("nodeHarvest")  # NodeHarvest: need to access raw file
install.packages("C:/study/R/_removed_packages/nodeHarvest_0.7-3.tar.gz", 
                 repos=NULL, type="source")
install.packages("inTrees")       # inTrees
install.packages("sirus")         # SIRUS
install.packages("remotes")
remotes::install_github("massimoaria/e2tree") # E2Tree
install.packages("ranktreeEnsemble")          # RankTree

## attach libraries
library(readr)
library(tidyverse)
library(rsample)
library(randomForest)

library(caret)
library(xrf)
library(parsnip)
library(rules)
#library(rulefit)

library(nodeHarvest)

library(inTrees)

library(sirus)

library(e2tree)

library(ranktreeEnsemble)

############# 1. Loading the data #############

# train-test: 0.70-0.30

setwd("C:/study/data/tree")
spam <- read_csv("spam.csv")
#bio <- read_csv("biores.csv")
#plant <- read_csv("plants100.csv")


## Simplify data (spam)
spam <- spam[,c(1:48, 58)]
spam <- as.data.frame(spam)
spam$class <- factor(spam$class)

set.seed(12345)
spam.split <- spam %>% initial_split(prop=0.75)
spam.train <- spam.split %>% training()
spam.test <- spam.split %>% testing()

x.spam.train <- spam.train %>% select(-class)
x.spam.test <- spam.test %>% select(-class)

target.spam.train <- factor(spam.train$class)
target.spam.test <- factor(spam.test$class)


############# 2. Fit to the model #############

## random forest (baseline, not rule-extraction method)
set.seed(123)
start.time <- Sys.time()
spam.class <- randomForest(factor(class) ~ ., data=spam.train, importance=TRUE)
end.time <- Sys.time()
total.time <- end.time - start.time # 11.85315 secs

confusionMatrix(spam.class$predicted, reference=target.spam.train)  # Train Accuracy : 0.9386

spam.test.pred <- predict(spam.class, newdata=x.spam.test)
confusionMatrix(spam.test.pred, reference=target.spam.test)       # Test Accuracy : 0.9461


## RuleFit
set.seed(123)
start.time <- Sys.time()
spam.class.rule <- rule_fit(mode="classification") %>% fit(factor(class) ~ ., data=spam.train)
#spam.class.rule <- gbm.fit(x=x.spam.train, y=target.spam.train) %>% rulefit(n.trees=500)
end.time <- Sys.time()
total.time <- end.time - start.time # 15.06144 secs

spam.train.rule.pred <- predict(spam.class.rule, new_data=x.spam.train) %>% factor()
confusionMatrix(spam.train.rule.pred$.pred_class, reference=target.spam.train)  # Train Accuracy : 0.8803

spam.test.rule.pred <- stats::predict(spam.class.rule, new_data=x.spam.test)
confusionMatrix(spam.test.rule.pred$.pred_class, reference=target.spam.test)    # Test Accuracy : 0.8862

spam.class.rule %>% tidy(penalty=0.1) # readable rules (321 rules)


## NodeHarvest
set.seed(123)
start.time <- Sys.time()
spam.class.harv <- nodeHarvest(x.spam.train, as.numeric(target.spam.train)-1)   # target should be numeric?
end.time <- Sys.time()
total.time <- end.time - start.time # 24.71917 secs

spam.train.harv.pred <- factor(round(spam.class.harv$predicted))
confusionMatrix(spam.train.harv.pred, reference=target.spam.train)  # Train Accuracy : 0.8777 

spam.test.harv.pred <- stats::predict(spam.class.harv, newdata=x.spam.test) %>% round() %>% factor()
confusionMatrix(spam.test.harv.pred, reference=target.spam.test)    # Test Accuracy : 0.8905 

plot(spam.class.harv) # readable rules (54 rules)
print(spam.class.harv, nonodes=10)


## inTrees
set.seed(123)
start.time <- Sys.time()  # only time for rule extraction, random forest model used
intree.lists <- spam.class %>% RF2List()
intree.rules <- extractRules(intree.lists, x.spam.train)
intree.rule.metric <- getRuleMetric(intree.rules, x.spam.train, target.spam.train)
intree.rule.metric <- pruneRule(intree.rule.metric, x.spam.train, target.spam.train)
spam.class.intree <- buildLearner(intree.rule.metric, x.spam.train, target.spam.train)
end.time <- Sys.time()
total.time <- end.time - start.time # 19.96881 secs

spam.train.intree.pred <- applyLearner(spam.class.intree, x.spam.train) %>% factor()
confusionMatrix(spam.train.intree.pred, reference=target.spam.train)  # Train Accuracy : 0.9267

spam.test.intree.pred <- applyLearner(spam.class.intree, x.spam.test) %>% factor()
confusionMatrix(spam.test.intree.pred, reference=target.spam.test)    # Test Accuracy : 0.9201

presentRules(spam.class.intree, colnames(x.spam.train))  # readable rules (3463 rules)


## SIRUS
set.seed(123)
start.time <- Sys.time()
spam.class.sirus <- sirus.fit(x.spam.train, as.numeric(target.spam.train)-1, # designed for <= 100 rules
                              type="classif", num.rule=10)  # need numeric targets... only takes binary
end.time <- Sys.time()
total.time <- end.time - start.time # 1.32356 secs

spam.train.sirus.pred <- sirus.predict(spam.class.sirus, x.spam.train) %>% round() %>% factor()
confusionMatrix(spam.train.sirus.pred, reference=target.spam.train) # Train Accuracy : 0.8046 

spam.test.sirus.pred <- sirus.predict(spam.class.sirus, x.spam.test) %>% round() %>% factor()
confusionMatrix(spam.test.sirus.pred, reference=target.spam.test) # Train Accuracy : 0.8063

sirus.print(spam.class.sirus) # readable rules (10 rules)


## E2Tree
set.seed(123)
spam.ensemble <- randomForest(factor(class) ~ ., data=spam.train, importance=TRUE, proximity=TRUE)
#D <- createDisMatrix(spam.ensemble, data=spam.train, label="class", parallel=TRUE)
# takes too much time, not appropriate for large dataset?

start.time <- Sys.time()
spam.class.e2tree <- e2tree(factor(class) ~ ., data=spam.train, 
                            D=spam.ensemble$proximity, ensemble=spam.ensemble)
end.time <- Sys.time()
total.time <- end.time - start.time # 0.9419708 secs

rpart2Tree(spam.class.e2tree, spam.ensemble) %>% rpart.plot::rpart.plot() # ?


## RankTree
set.seed(123)
spam.rank.ensemble <- rforest(class ~ ., data=spam.train, importance=TRUE, dimreduce=FALSE)
# dimension reduction option: rule extraction speed up, accuracy down?

start.time <- Sys.time()  # time for rule extraction
spam.class.rank <- extract.rules(spam.rank.ensemble)
end.time <- Sys.time()
total.time <- end.time - start.time # 9.33792 secs

spam.train.rank.pred <- ranktreeEnsemble::predict(spam.rank.ensemble, newdata=x.spam.train)
confusionMatrix(factor(spam.train.rank.pred$label), target.spam.train)  # Train Accuracy : 0.9794 

spam.test.rank.pred <- ranktreeEnsemble::predict(spam.rank.ensemble, newdata=x.spam.test)
confusionMatrix(factor(spam.test.rank.pred$label), target.spam.test)  # Test Accuracy : 0.9348

spam.class.rank$rule  # readable rules (20 rules)
