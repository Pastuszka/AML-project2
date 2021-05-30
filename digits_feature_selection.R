options(java.parameters = "-Xmx8000m")
source("WB_feature_selection.R")
source("MP_feature_selection.R")
source('test_models.R')

load("data/digits.rds")
colnames(digits_train_labels) <- "y"
digits <- cbind(digits_train, y=as.factor(digits_train_labels$y))



# Random Forest feature importance
model_rf_dgt <- randomForest(y~., data = digits, importance=TRUE)
varImpPlot(model_rf_dgt)
df_gini_dgt <- as.data.frame(model_rf_dgt$importance) %>% select(MeanDecreaseAccuracy) %>% arrange(-MeanDecreaseAccuracy)
barplot(df_gini_dgt$MeanDecreaseAccuracy[1:200], names.arg = rownames(df_gini_dgt[1:200,]))
vi_rf_gini_dgt <- cutoff.k(as.data.frame(model_rf_dgt$importance) %>% select(MeanDecreaseGini) %>% arrange(-MeanDecreaseGini),k=11)
vi_rf_acc_dgt <-  cutoff.k(as.data.frame(model_rf_dgt$importance) %>% select(MeanDecreaseAccuracy) %>% arrange(-MeanDecreaseAccuracy), k=11)

vi_ran_imp_dgt <- var_imp_ranger_impurity(digits, biggest_diff = FALSE, n = 20)
vi_ran_perm_dgt <- var_imp_ranger_perm(digits)

corrplot::corrplot(cor(as.matrix(digits[, vi_ran_imp_dgt])), method = "number")
corrplot::corrplot(cor(as.matrix(digits[, vi_ran_imp_dgt])), method = "square")
# chi square test
vi_chisq_dgt <- var_imp_chisq(digits)

# entropy-based
  
weights_dgt <- gain.ratio(y~., data=digits)
weights_dgt <- weights_dgt %>% arrange(-attr_importance) %>% filter(attr_importance>0)
barplot(weights_dgt$attr_importance[1:17])
vi_gain_rat_dgt2 <- rownames(weights_dgt)[1:17]
entropy_fs_models_dgt <- test_all_models(digits[, c(vi_gain_rat_dgt2, "y")])


# lasso feature selection -------------------------------------------------


vi_lasso_dgt <- var_imp_lasso(digits)
vi_lasso_1se_dgt <- var_imp_lasso(digits)


# MRMR
library(praznik)
vi_mrmr <- MRMR(X=digits %>% select(-y), Y=digits$y, k=50)
vi_cmim <- CMIM(X=digits %>% select(-y), Y=digits$y, k=50)
vi_jmi <- JMI(X=digits %>% select(-y), Y=digits$y, k=50)
vi_dsir <- DISR(X=digits %>% select(-y), Y=digits$y, k=50)
vi_jmim <- JMIM(X=digits %>% select(-y), Y=digits$y, k=50)
vi_njmim <- NJMIM(X=digits %>% select(-y), Y=digits$y, k=50)

selected_features_dgt <- c()
for (vi in list(vi_mrmr, vi_cmim, vi_jmi, vi_dsir, vi_jmim, vi_njmim)){
  selected_features_dgt <- c(selected_features_dgt,names(vi$selection))
}
vi_praznik_dgt <- names(sort(table(selected_features_dgt), decreasing = TRUE)[1:50])
corrplot::corrplot(cor(as.matrix(digits[, vi_praznik_dgt])), method = "number")
corrplot::corrplot(cor(as.matrix(digits[, vi_praznik_dgt])), method = "square")

praznik_fs_models_dgt <- test_all_models(digits[, c(vi_praznik_dgt, "y")])
