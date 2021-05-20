library(glmnet)
library(xgboost)
library(DALEX)
library(Boruta)
library(ranger)
# Lasso variable selection ----------------------------------------------

var_imp_lasso <- function(data, lambda.1se=FALSE){
  X <- as.matrix(subset(data, select = -c(y)))
  y <- as.matrix(subset(data, select = c(y)))
  cv.lasso <- cv.glmnet(X, y, family="binomial", alpha=1)
  lambda <- ifelse(lambda.1se, cv.lasso$lambda.1se, cv.lasso$lambda.min)
  model <- glmnet(X, y, family="binomial", alpha=1, lambda = lambda)
  colnames(X[,which(model$beta!=0)])
}

# XGBoost feature importance ----------------------------------------------

var_imp_caret <- function(data, model="xgbTree", n=10, biggest_diff=TRUE, ...){
  model <- train(y~., data=data, method="xgbTree", ...)
  var_imp <- varImp(model)$importance
  if (biggest_diff) return(cutoff.biggest.diff(var_imp))
  else return(cutoff.k(var_imp,n))
}

# DALEX feature importance ----------------------------------------------

var_imp_DALEX <- function(data, n=10, biggest_diff=TRUE){
  X <- subset(data, select = -c(y))
  dat <- data
  dat$y <- dat$y == '1'
  model_rf <- ranger(y~., data=dat)
  explained_model <- explain(model_rf, data=X, y=dat$y)
  varimps <- feature_importance(explained_model)
  #DALEX is the most broken piece of dunskie warchlaki i've seen
  var_imp <- data.frame(importance = numeric(), column = character())
  for(i in 1:ncol(X)){
    var_imp[i, 'importance'] <- varimps[i+1, 2]
    var_imp[i, 'column'] <- varimps[i+1, 1]
  }
  rownames(var_imp) <- var_imp$column
  if (biggest_diff) return(cutoff.biggest.diff(var_imp))
  else return(cutoff.k(var_imp,n))
}

# Correlation feature importance ----------------------------------------------

var_imp_corr <- function(data, type='pearson', n=10, biggest_diff=TRUE){
  X <- subset(data, select = -c(y))
  y <- subset(data, select = c(y))
  var_imp <- abs(cor(X, y = y, method = type))
  if (biggest_diff) return(cutoff.biggest.diff(var_imp))
  else return(cutoff.k(var_imp,n))
}

# Boruta feature importance ----------------------------------------------

var_imp_boruta <- function(data, rough_fix=TRUE, with_tentative=FALSE){
  boruta_output <- Boruta(y~., data=data)
  if(rough_fix) boruta_output <- TentativeRoughFix(boruta_output)
  getSelectedAttributes(boruta_output, withTentative = with_tentative)
}

# Genetic feature importance ----------------------------------------------

var_imp_genetic <- function(data, iters=100, really_do_it=FALSE){
  if(really_do_it){
    X <- subset(data, select = -c(y))
    y <- data$y
    ga_ctrl <- gafsControl(functions = rfGA,  # another option is `caretGA`.
                           method = "cv",
                           repeats = 3)
    ga_obj <- gafs(x=X, 
                   y=y, 
                   iters = iters,   # normally much higher (100+)
                   gafsControl = ga_ctrl)
    return(ga_obj$optVariables)
  }else{
    print('Don\'t even try')
    return(character())
  }
  
}

# Information value ----------------------------------------------

# used for categorical variables, so useless