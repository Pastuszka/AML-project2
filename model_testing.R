source('test_models.R')
source("utils.R")
library(dplyr)
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


# digits - ensemble of feature selection - classification --------------------------


source("digits_feature_selection.R")

selected_features_dgt <- c(selected_features_dgt, vi_rf_gini_dgt, vi_rf_acc_dgt, vi_ran_imp_dgt, vi_ran_perm_dgt, vi_chisq_dgt, vi_lasso_dgt, vi_lasso_1se_dgt)

sorted_features <- sort(table(selected_features_dgt), decreasing = TRUE)

top_features <- names(sorted_features)[which(sorted_features>2)]

subset_digits1 <- digits[, c(top_features, "y")]
digits_models1 <- test_all_models(subset_digits1)

corrplot::corrplot(cor(as.matrix(digits[, top_features])), method = "number", number.cex=0.5)
corrplot::corrplot(cor(as.matrix(digits[, top_features])), method = "square")

subset_digits3 <- digits[, c(top_features, "y")] %>% select(-V339, -V4413)
digits_models3 <- test_all_models(subset_digits3)


# artificial - ensemble of feature selection   -------------------------------------------------------------
source("artif_feature_selection.R")

selected_features_artif <- c(praznik_cols, vi_rf_gini_artif, vi_rf_acc_artif, vi_ran_imp_artif, vi_ran_perm_artif, vi_chisq_artif, vi_lasso_artif, vi_lasso_1se_artif)

sorted_features_artif <- sort(table(selected_features_artif), decreasing = TRUE)

top_features_artif <- names(sorted_features_artif)[which(sorted_features_artif>4)]

subset_artif2<- artif[, c(top_features_artif, "y")]
artif_models2 <- test_all_models(subset_artif2)
