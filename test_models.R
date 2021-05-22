library(mlr3verse)
library(mlrintermbo)

test_model <- function(dataset, learner, search_space){
  
  task <- TaskClassif$new(id = "test_task", backend = dataset, target = "y")
  
  terminator = trm("evals", n_evals = 50)
  
  rdesc = rsmp("cv", folds = 5)
  
  measure <- msr("classif.bacc")
  
  instance = TuningInstanceSingleCrit$new(
    task = task,
    learner = learner,
    resampling = rdesc,
    measure = measure,
    search_space = search_space,
    terminator = terminator
  )
  
  tuner <- tnr("intermbo")
  
  tuner$optimize(instance)
  
  instance
}

test_ranger <- function(dataset){
  
  learner <- lrn("classif.ranger")
  
  search_space = ps(
    mtry = p_int(lower = 1, upper  = ncol(dataset)-1),
    min.node.size = p_int(lower = 1, upper = 100),
    max.depth = p_int(lower = 1, upper = 100)
  )
  
  test_model(dataset, learner, search_space)
}

test_xgboost <- function(dataset){
  
  learner <- lrn("classif.xgboost")
  
  search_space = ps(
    nrounds = p_int(lower = 1, upper  = 500),
    eta = p_dbl(lower = 0.01, upper = 1),
    gamma = p_dbl(lower = 0, upper = 10),
    max_depth = p_int(lower = 1, upper = 100),
    min_child_weight = p_dbl(lower = 0, upper = 10)
  )
  
  test_model(dataset, learner, search_space)
}

test_xgboost <- function(dataset){
  
  learner <- lrn("classif.xgboost")
  
  search_space = ps(
    nrounds = p_int(lower = 100, upper  = 500),
    eta = p_dbl(lower = 0.01, upper = 1),
    gamma = p_dbl(lower = 0, upper = 10),
    max_depth = p_int(lower = 1, upper = 100),
    min_child_weight = p_dbl(lower = 0, upper = 10)
  )
  
  test_model(dataset, learner, search_space)
}

test_rpart <- function(dataset){
  
  learner <- lrn("classif.rpart")
  
  search_space = ps(
    minsplit = p_int(lower = 1, upper  = 100),
    minbucket = p_int(lower = 1, upper = 100),
    cp = p_dbl(lower = 0.001, upper = 0.1)
  )
  
  test_model(dataset, learner, search_space)
}

test_svm <- function(dataset){
  
  learner <- lrn("classif.svm", type='C-classification')
  
  search_space = ps(
    cost = p_int(lower = 1, upper  = 10),
    kernel = p_fct(levels = c('radial', 'linear', 'polynomial', 'sigmoid'))
  )
  
  test_model(dataset, learner, search_space)
}

test_nnet <- function(dataset){
  
  learner <- lrn("classif.nnet")
  
  search_space = ps(
    size = p_int(lower = 1, upper  = 100),
    skip = p_lgl()
  )
  
  test_model(dataset, learner, search_space)
}

test_kknn <- function(dataset){
  
  learner <- lrn("classif.kknn")
  
  search_space = ps(
    k = p_int(lower = 1, upper  = 20),
    kernel = p_fct(levels = c('rectangular', "triangular", "epanechnikov", "biweight",
                              "triweight", "cos", "inv", "gaussian"))
  )
  
  test_model(dataset, learner, search_space) 
}

test_lda <- function(dataset){
  
  learner <- lrn("classif.lda")
  
  search_space = ps(
    method = p_fct(levels = c('moment', "mle", "mve"))
  )
  
  test_model(dataset, learner, search_space) 
}

test_qda <- function(dataset){
  
  learner <- lrn("classif.qda")
  
  search_space = ps(
    method = p_fct(levels = c('moment', "mle", "mve"))
  )
  
  test_model(dataset, learner, search_space) 
}

test_glm <- function(dataset){
  
  learner <- lrn("classif.glmnet")
  
  search_space = ps(
    alpha = p_dbl(lower=0, upper=1)
  )
  
  test_model(dataset, learner, search_space) 
}

test_naive_bayes <- function(dataset){
  
  learner <- lrn("classif.naive_bayes")
  
  search_space = ps(
    laplace = p_dbl(lower=0, upper=100)
  )
  
  test_model(dataset, learner, search_space) 
}
