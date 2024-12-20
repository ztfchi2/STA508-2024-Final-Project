
library(caret)
library(rpart)
library(visNetwork)

data(diamonds)
head(diamonds)

sim.dia <- diamonds
head(sim.dia)

set.seed(345)
dia.train <- createDataPartition(y=sim.dia$cut, p=0.75, list=FALSE)
train <- sim.dia[dia.train,]
test <- sim.dia[-dia.train,]

d.tree <- rpart(cut ~ ., data=sim.dia)
d.tree
visTree(d.tree, legend=FALSE) # Figure 1
