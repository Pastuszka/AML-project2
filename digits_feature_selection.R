options(java.parameters = "-Xmx8000m")
source("WB_feature_selection.R")
source("MP_feature_selection.R")

load("data/digits.rds")
colnames(digits_train_labels) <- "y"
digits <- cbind(digits_train, y=as.factor(digits_train_labels$y))


# Bagging feature importance
# too slow
# vi_bagging_dgt <- var_imp_bagging(digits)

# Random Forest feature importance
model_rf_dgt <- randomForest(y~., data = digits, importance=TRUE)
varImpPlot(model_rf_dgt)
vi_rf_gini_dgt <- cutoff.biggest.diff(as.data.frame(model_rf_dgt$importance) %>% select(MeanDecreaseGini) %>% arrange(-MeanDecreaseGini))
vi_rf_acc_dgt <-  cutoff.biggest.diff(as.data.frame(model_rf_dgt$importance) %>% select(MeanDecreaseAccuracy) %>% arrange(-MeanDecreaseAccuracy))

vi_ran_imp_dgt <- var_imp_ranger_impurity(digits)
vi_ran_perm_dgt <- var_imp_ranger_perm(digits)

# chi square test
vi_chisq_dgt <- var_imp_chisq(digits)

# entropy-based
  

# Recursive feature elimination
# too slow
# vi_rfe_dgt <- var_imp_rfe(digits)

# AIC bIC


# MCFS
# doesn't work
# somehow changes working directory
# actually works but not on whole dataset
# vi_mcfs1_dgt <- var_imp_mcfs(digits[,c(1:400, 4956)])
# vi_mcfs2_dgt <- var_imp_mcfs(artif[,c(401:500, 501)])
# vi_mcfs <- c(vi_mcfs1, vi_mcfs2)

vi_lasso_dgt <- var_imp_lasso(digits)
vi_lasso_1se_dgt <- var_imp_lasso(digits)

# Caret feature importance
#slow
# vi_caret_dgt <- var_imp_caret(digits)

#DALEX feature importance-srance
#slow but that's fine because it doesn't work anyway
# vi_dalex <- var_imp_DALEX(artif) #broken dunskie warchlaki doesn't work

#Correlation feature importance
# vi_corr_dgt <- var_imp_corr(digits[1:1000,])

#Boruta feature importance
#kinda slow
# vi_boruta_dgt <- var_imp_boruta(digits)

# MRMR
library(praznik)
vi_mrmr <- MRMR(X=digits %>% select(-y), Y=digits$y, k=50)
vi_cmim <- CMIM(X=digits %>% select(-y), Y=digits$y, k=50)
vi_jmi <- JMI(X=digits %>% select(-y), Y=digits$y, k=50)
vi_dsir <- DISR(X=digits %>% select(-y), Y=digits$y, k=50)
vi_jmim <- JMIM(X=digits %>% select(-y), Y=digits$y, k=50)
vi_njmim <- NJMIM(X=digits %>% select(-y), Y=digits$y, k=50)

selected_features_dgt <- c()
for (vi in list(vi_mrmr, vi_cmim, vi_jmi, vi_dsir, vi_dsir, vi_jmim, vi_njmim)){
  selected_features_dgt <- c(selected_features_dgt,names(vi$selection[1:15]))
}
sort(table(selected_features_dgt), decreasing = TRUE)
