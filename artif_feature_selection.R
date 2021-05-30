options(java.parameters = "-Xmx8000m")
source("WB_feature_selection.R")
source("MP_feature_selection.R")
source('test_models.R')

load("data/artificial.rds")
colnames(artif_train_labels) <- "y"
artif <- cbind(artif_train, y=as.factor(artif_train_labels$y))



# Random Forest based feature importance ----------------------------------------
set.seed(1235)
model_rf_artif <- randomForest(y~., data = artif, importance=TRUE)
varImpPlot(model_rf_artif)
vi_rf_gini_artif  <- cutoff.k(as.data.frame(model_rf_artif$importance) %>% select(MeanDecreaseGini) %>% arrange(-MeanDecreaseGini), k=19)
vi_rf_acc_artif <-  cutoff.k(as.data.frame(model_rf_artif$importance) %>% select(MeanDecreaseAccuracy) %>% arrange(-MeanDecreaseAccuracy), k=19)

# ranger
vi_ran_imp_artif  <- var_imp_ranger_impurity(artif, biggest_diff = FALSE, n=19)
vi_ran_perm_artif <- var_imp_ranger_perm(artif, biggest_diff = FALSE, n=19)

corrplot::corrplot(cor(as.matrix(artif[, vi_ran_perm_artif])), method = "number")
corrplot::corrplot(cor(as.matrix(artif[, vi_ran_perm_artif])), method = "square")
table(c(vi_rf_gini_artif, vi_rf_acc_artif, vi_ran_imp_artif, vi_ran_perm_artif, c("V242", "V476", "V339", "V106", "V49", "V379", "V129", "V337", "V473", "V282", "V443", "V154", "V434", "V454", "V494", "V65", "V29", "V452", "V319")))

rf_fs_models <- test_all_models(artif[, c(vi_ran_perm_artif, "y")])


# chi square test ---------------------------------------------------------

vi_chisq_artif <- var_imp_chisq(artif)


# lasso -------------------------------------------------------------------

vi_lasso_artif <- var_imp_lasso(artif)
vi_lasso_1se_artif <- var_imp_lasso(artif)

# entropy-based -----------------------------------------------------------

weights_artif <- gain.ratio(y~., data=artif)
weights_artif <- weights_artif %>% arrange(-attr_importance) %>% filter(attr_importance>0)
barplot(weights_artif$attr_importance)

vi_inf_gain_artif <- var_imp_inf_gain(artif, biggest_diff = FALSE, n=13)
vi_gain_rat_artif <- var_imp_gain_rat(artif, biggest_diff = FALSE, n=13)
vi_sym_unc_artif <- var_imp_sym_unc(artif, biggest_diff = FALSE, n=13)

entropy_cols <- names(sort(table(c(vi_inf_gain_artif, vi_gain_rat_artif, vi_sym_unc_artif)), decreasing = TRUE))
corrplot::corrplot(cor(as.matrix(artif[, entropy_cols])), method = "number")
corrplot::corrplot(cor(as.matrix(artif[, entropy_cols])), method = "square")

entropy_fs_models <- test_all_models(artif[, c(entropy_cols, "y")])


# Recursive feature elimination

vi_rfe <- var_imp_rfe(artif)
corrplot::corrplot(cor(as.matrix(artif[, vi_rfe])), method = "number")
corrplot::corrplot(cor(as.matrix(artif[, vi_rfe])), method = "square")
funModeling::plot_num(artif[, c(476, 242)], bins = 50)
funModeling::cross_plot(artif[, c("V476", "V242","y")], target = "y")
funModeling::plotar(artif[, c("V476", "V242","y")], target = "y", plot_type = "histdens")
funModeling::plot_num(artif[, c(49, 379)], bins = 50)
funModeling::cross_plot(artif[, c("V49", "V379","y")], target = "y")
funModeling::plotar(artif[, c("V49", "V379","y")], target = "y", plot_type = "histdens")



#Boruta feature importance
# vi_boruta_artif <- var_imp_boruta(artif)

# MRMR
library(praznik)
vi_mrmr_artif <- MRMR(X=artif %>% select(-y), Y=artif$y, k=40)
vi_cmim_artif <- CMIM(X=artif %>% select(-y), Y=artif$y, k=40)
vi_jmi_artif <- JMI(X=artif %>% select(-y), Y=artif$y, k=40)
vi_dsir_artif <- DISR(X=artif %>% select(-y), Y=artif$y, k=40)
vi_jmim_artif <- JMIM(X=artif %>% select(-y), Y=artif$y, k=40)
vi_njmim_artif <- NJMIM(X=artif %>% select(-y), Y=artif$y, k=40)

selected_features_artif <- c()
for (vi in list(vi_mrmr_artif, vi_cmim_artif, vi_jmi_artif, vi_dsir_artif, vi_jmim_artif, vi_njmim_artif)){
  selected_features_artif <- c(selected_features_artif,names(vi$selection))
}
praznik_cols <- names(sort(table(selected_features_artif), decreasing = TRUE)[1:22])
corrplot::corrplot(cor(as.matrix(artif[, praznik_cols])), method = "number")
corrplot::corrplot(cor(as.matrix(artif[, praznik_cols])), method = "square")
corrplot::corrplot(cor(as.matrix(artif[, names(vi_mrmr_artif$selection)])), method = "square")

ranger_praznik <- test_ranger(artif[, c(praznik_cols, "y")])
ranger_mrmr <- test_ranger(artif[, c(vi_mrmr_artif[1:20], "y")])
glm_praznik <- test_glm(artif[, c(praznik_cols, "y")])
svm_praznik <- test_svm(artif[, c(praznik_cols, "y")])

vi_mrmr_artif2 <- MRMR(X=artif %>% select(vi_boruta), Y=artif$y, k=10)
corrplot::corrplot(cor(as.matrix(artif[, names(vi_mrmr_artif2$selection)])), method = "square")
corrplot::corrplot(cor(as.matrix(artif[,c("V29",  "V49",  "V65",  "V106", "V154", "V242", "V339","V443", "V454", "V456")])), method = "square")
