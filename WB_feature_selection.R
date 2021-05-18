library(adabag)
library(randomForest)
library(FSelector)
library(caret)
library(dplyr)
library(rmcfs)

# Bagging feature importance ----------------------------------------------

var_imp_bagging <- function(data, n=10, biggest_diff=TRUE){
  model <- bagging(y~., data = data, mfinal = 100)
  var_imp <- data.frame(sort(model$importance, decreasing = TRUE))
  if (biggest_diff) return(cutoff.biggest.diff(var_imp))
  else return(cutoff.k(var_imp,n))
}


# Random Forest feature importance ----------------------------------------

var_imp_rf_acc <- function(data, n=10, biggest_diff=TRUE){
  model <- randomForest(y~., data = data, importance=TRUE)
  var_imp_acc <- as.data.frame(model$importance) %>% select(MeanDecreaseAccuracy) %>% arrange(-MeanDecreaseAccuracy)
  if (biggest_diff) return(cutoff.biggest.diff(var_imp_acc))
  else return(cutoff.k(var_imp_acc,n))
}

var_imp_rf_gini <- function(data, n=10, biggest_diff=TRUE){
  model <- randomForest(y~., data = data, importance=TRUE)
  var_imp_gini <- as.data.frame(model$importance) %>% select(MeanDecreaseGini) %>% arrange(-MeanDecreaseGini)
  if (biggest_diff) return(cutoff.biggest.diff(var_imp_gini))
  else return(cutoff.k(var_imp_gini,n))
}


# chi square test ---------------------------------------------------------

var_imp_chisq <- function(data, n=10, biggest_diff=TRUE){
  weights <- chi.squared(y~., data=data)
  if (biggest_diff) return(cutoff.biggest.diff(weights))
  else return(cutoff.k(weights,n))
}

# consistency-based --------------------------------------------------------

# cons <- consistency(y~., data=artif) # slow


# entropy-based -----------------------------------------------------------

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

# very slow
# Define control function
# sa_ctrl <- safsControl(functions = rfSA,
#                        method = "repeatedcv",
#                        number=5,
#                        repeats = 2,
#                        improve = 5, verbose=TRUE) # n iterations without improvement before a reset
# 
# # Genetic Algorithm feature selection
# set.seed(100)
# sa_obj <- safs(x=artif %>% select(-y), 
#                y=artif$y,
#                safsControl = sa_ctrl,
#                iters = 10)
# 
# sa_obj$optVariables
# plot(sa_obj)
# AIC BIC -----------------------------------------------------------------
var_imp_AIC <- function(data){
  data$y <- ifelse(data$y>0, 1, 0)
  base_model <- glm(y~1, data=data, family="binomial")
  full_model <- glm(y~., data=data, family="binomial")
  step_AIC <- step(base_model, scope = list(lower = base_model, upper = full_model), direction = "both", trace = 1, steps = 1000, k=2)
  return(names(step_AIC$coefficients)[-1])
}

var_imp_BIC <- function(data){
  data$y <- ifelse(data$y>0, 1, 0)
  base_model <- glm(y~1, data=data, family="binomial")
  full_model <- glm(y~., data=data, family="binomial")
  step_BIC <- step(base_model, scope = list(lower = base_model, upper = full_model), direction = "both", trace = 1, steps = 1000, k=nrow(data))
  return(names(step_AIC$coefficients)[-1])
}


# MCFS --------------------------------------------------------------------


var_imp_mcfs <- function(data){
  model_mcfs <- mcfs(y~., data=data, cutoffPermutations=0)
  return(model_mcfs$RI$attribute[1:model_mcfs$cutoff_value])
}

