source('test_models.R')
source("utils.R")

# artificial --------------------------------------------------------------
load("data/artificial.rds")
colnames(artif_train_labels) <- "y"
artif <- cbind(artif_train, y=as.factor(artif_train_labels$y))



columns <- c("V476", "V242", "V339", "V49",  "V106", "V129", "V443", "V379", "V473", "V337", "V65",  "V454", "V319", "V494", 'y')

subset_artif <- artif[,columns]

test_ranger(subset_artif)
test_xgboost(subset_artif)
test_rpart(subset_artif)
test_svm(subset_artif)

test_nnet(subset_artif) # sounds good, doesn't work (literally)
optim_kknn <- test_kknn(subset_artif) # good results
test_lda(subset_artif) # poor results
test_qda(subset_artif) # poor results
test_glm(subset_artif) # poor results
test_naive_bayes(subset_artif) # poor results


test_all_models <- function(dataset){
  results <- list()
  scores <- data.frame(bacc=numeric())
  
  results$ranger <- test_ranger(dataset)
  scores['ranger',] <- results$ranger$result$classif.bacc
  results$xgboost <- test_xgboost(dataset)
  scores['xgboost',] <- results$xgboost$result$classif.bacc
  results$rpart <- test_rpart(dataset)
  scores['rpart',] <- results$rpart$result$classif.bacc
  results$svm <- test_svm(dataset)
  scores['svm',] <- results$svm$result$classif.bacc
  results$kknn <- test_kknn(dataset)
  scores['kknn',] <- results$kknn$result$classif.bacc
  
  results$scores <- scores
  print(scores)
  results
}

col_boruta <- c("V29",  "V49",  "V65",  "V106", "V129", "V154", "V242", "V282",
                "V319", "V337", "V339", "V379", "V434", "V443", "V452", "V454",
                "V456", "V473", "V476", "V494", 'y')

subset_artif_boruta <- artif[,col_boruta]

artif_boruta_models <- test_all_models(subset_artif_boruta)

corrplot::corrplot(cor(as.matrix(subset(subset_artif_boruta, select=-c(y)))))

col_boruta_decor <- c("V29",  "V49",  "V65",  "V106", "V154",
                      #"V242",
                "V339", "V443", "V454",
                "V456", 'y')

subset_artif_boruta_decor <- artif[,col_boruta_decor]
corrplot::corrplot(cor(as.matrix(subset(subset_artif_boruta_decor, select=-c(y)))))

artif_boruta_decor_models <- test_all_models(subset_artif_boruta_decor)

col_boruta_decor2 <- c("V29",  "V49",  "V65",  "V106", "V154",
                       "V443", "V454",
                      "V456", 'y')

subset_artif_boruta_decor2 <- artif[,col_boruta_decor2]
corrplot::corrplot(cor(as.matrix(subset(subset_artif_boruta_decor2, select=-c(y)))))

artif_boruta_decor2_models <- test_all_models(subset_artif_boruta_decor2)

col_boruta_decor3 <- c("V29",  "V49",  "V65",  "V106", "V154",
                       "V443", "V454",
                       #"V456",
                       'y')

subset_artif_boruta_decor3 <- artif[,col_boruta_decor3]
corrplot::corrplot(cor(as.matrix(subset(subset_artif_boruta_decor3, select=-c(y)))))

artif_boruta_decor3_models <- test_all_models(subset_artif_boruta_decor3)


col_boruta_decor4 <- c("V29",  "V49",  "V65",  "V106", "V154",
                        "V454",
                       
                       'y')

subset_artif_boruta_decor4 <- artif[,col_boruta_decor4]
corrplot::corrplot(cor(as.matrix(subset(subset_artif_boruta_decor4, select=-c(y)))))

artif_boruta_decor4_models <- test_all_models(subset_artif_boruta_decor4)

drop_1_var_knn <- function(dataset){
  vars <- colnames(dataset)
  accuracies <- data.frame(accuracy=numeric())
  for(var in vars){
    if(var == 'y') next
    sub_dataset <- dataset[ , -which(names(dataset) %in% c(var))]
    test_result <- test_kknn(sub_dataset)
    accuracies[var,] <- test_result$result$classif.bacc
  }
  accuracies
}

drop1 <- drop_1_var_knn(subset_artif_boruta)

greedy_knn_selection <- function(dataset){
  steps <- data.frame(dropped = character(), score = numeric())
  best_score <- -Inf
  while(ncol(dataset) > 1){
    result <- drop_1_var_knn(dataset)
    max_result <- which.max(result$accuracy)
    cur_score <- result[max_result,'accuracy']
    cur_col <- rownames(result)[max_result]
    if(cur_score > best_score){
      best_score <- cur_score
      dataset <- dataset[ , -which(names(dataset) %in% c(cur_col))]
      steps[length(steps)+1, 'dropped'] <- cur_col
      steps[length(steps)+1, 'score'] <- cur_score
    }else{
      break
    }
  }
  steps
}

col_boruta_drop1 <- c( "V49",  "V65",  "V106", "V129", "V154", "V242", "V282",
                       "V319", "V337", "V339", "V379", "V434", "V443", "V452", "V454",
                      "V456", "V473", "V476", "V494", 'y')

subset_boruta_drop1 <- artif[,col_boruta_drop1]

greedy_result <- greedy_knn_selection(subset_boruta_drop1) # dropped v129

 # digits ------------------------------------------------------------------

load("data/digits.rds")
colnames(digits_train_labels) <- "y"
digits <- cbind(digits_train, y=as.factor(digits_train_labels$y))

imp_columns <- get_important_columns(digits)

subset_digits <- digits[, imp_columns$selected_cols]

digits_boruta <- var_imp_boruta(digits)

save(digits_boruta, file='digits_boruta.rds')
load("digits_boruta.rds")

col_digits_boruta <- c(digits_boruta, 'y')

subset_digits_boruta <- digits[,col_digits_boruta]

digits_boruta_models <- test_all_models(subset_digits_boruta)

source("digits_feature_selection.R")

selected_features_dgt <- c(selected_features_dgt, vi_rf_gini_dgt, vi_rf_acc_dgt, vi_ran_imp_dgt, vi_ran_perm_dgt, vi_chisq_dgt, vi_lasso_dgt, vi_lasso_1se_dgt)

sorted_features <- sort(table(selected_features_dgt), decreasing = TRUE)

top_features <- names(sorted_features)[which(sorted_features>2)]

subset_digits1 <- digits[, c(top_features, "y")]
digits_models1 <- test_all_models(subset_digits1)
