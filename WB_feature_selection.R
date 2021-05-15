library(adabag)
library(randomForest)
library(FSelector)
library(caret)
library(dplyr)

# data preparation --------------------------------------------------------

load("data/artificial.rds")
colnames(artif_train_labels) <- "y"
artif <- cbind(artif_train, y=as.factor(artif_train_labels$y))
colnames(artif)

# Bagging feature importance ----------------------------------------------


model <- bagging(y~., data = artif, mfinal = 100)
var_imp <- sort(model$importance, decreasing = TRUE)

barplot(var_imp[1:40])
data.frame(var_imp)

var_imp_bagging <- function(data, n=10){
  model <- bagging(y~., data = data, mfinal = 100)
  var_imp <- sort(model$importance, decreasing = TRUE)
  return(names(var_imp[1:n]))
}


# Random Forest feature importance ----------------------------------------

model2 <- randomForest(y~., data = artif, importance=TRUE)
var_imp2 <- as.data.frame(model2$importance)
var_imp2_acc <- var_imp2[order(var_imp2$MeanDecreaseAccuracy, decreasing = TRUE),]
var_imp2_gini <- var_imp2[order(var_imp2$MeanDecreaseGini, decreasing = TRUE),]

barplot(var_imp2_acc[1:40,3])
barplot(var_imp2_gini[1:40,4])

var_imp_rf_acc <- function(data, n=10){
  model <- randomForest(y~., data = data, importance=TRUE)
  var_imp <- as.data.frame(model$importance)
  var_imp_acc <- var_imp[order(var_imp$MeanDecreaseAccuracy, decreasing = TRUE),]
  return(rownames(var_imp_acc)[1:n])
}

var_imp_rf_gini <- function(data, n=10){
  model <- randomForest(y~., data = data, importance=TRUE)
  var_imp <- as.data.frame(model$importance)
  var_imp_acc <- var_imp[order(var_imp$MeanDecreaseGini, decreasing = TRUE),]
  return(rownames(var_imp_acc)[1:n])
}


# chi square test ---------------------------------------------------------

weights <- chi.squared(y~., data=artif)

barplot(sort(weights$attr_importance, decreasing = TRUE)[1:20])
cutoff.biggest.diff(weights)
cutoff.k(weights,20)

var_imp_chisq <- function(data, n=10, biggest_diff=TRUE){
  weights <- chi.squared(y~., data=data)
  if (biggest_diff) return(cutoff.biggest.diff(weights))
  else return(cutoff.k(weights,n))
}


# consistency-based --------------------------------------------------------

cons <- consistency(y~., data=artif) # wolne


# entropy-based -----------------------------------------------------------
inf_gain <- information.gain(y~., data=artif)
cutoff.biggest.diff(inf_gain)
gain_rat <- gain.ratio(y~., data=artif)
cutoff.k(gain_rat, 10)
sym_unc <- symmetrical.uncertainty(y~., data=artif)
cutoff.k(sym_unc, 10)

var_imp_inf_gain <- function(data, n=10, biggest_diff=TRUE){
  weights <- information.gain(y~., data=data)
  if (biggest_diff) return(cutoff.biggest.diff(weights))
  else return(cutoff.k(weights,n))
}
var_imp_gain_rat <- function(data, n=10, biggest_diff=TRUE){
  weights <- gain.ratio(y~., data=data)
  if (biggest_diff) return(cutoff.biggest.diff(weights))
  else return(cutoff.k(weights,n))
}
var_imp_sym_unc<- function(data, n=10, biggest_diff=TRUE){
  weights <- symmetrical.uncertainty(y~., data=data)
  if (biggest_diff) return(cutoff.biggest.diff(weights))
  else return(cutoff.k(weights,n))
}


# Recursive feature elimination -------------------------------------------

# Define the control using a random forest selection function
control <- rfeControl(functions = rfFuncs, # random forest
                      method = "repeatedcv", # repeated cv
                      repeats = 3, # number of repeats
                      number = 5) # number of folds

# wolne
result_rfe1 <- rfe(x = artif_train, 
                   y = as.factor(artif_train_labels$y), 
                   sizes = c(5,10,15,20),
                   rfeControl = control)

predictors(result_rfe1)


var_imp_rfe <- function(data, n=10){
  control <- rfeControl(functions = rfFuncs, # random forest
                        method = "repeatedcv", # repeated cv
                        repeats = 3, # number of repeats
                        number = 5) # number of folds
  
  # wolne
  result_rfe1 <- rfe(x = data %>% select(-y), 
                     y = as.factor(data$y), 
                     sizes = c(n),
                     rfeControl = control)
  
  return(predictors(result_rfe1))
}


# simulated annealing -----------------------------------------------------

# Define control function
sa_ctrl <- safsControl(functions = rfSA,
                       method = "repeatedcv",
                       repeats = 3,
                       improve = 5) # n iterations without improvement before a reset

# Genetic Algorithm feature selection
set.seed(100)
sa_obj <- safs(x=artif[1:500,] %>% select(-y), 
               y=as.factor(artif$y[1:500]),
               safsControl = sa_ctrl,
               iters = 20)

sa_obj$optVariables
