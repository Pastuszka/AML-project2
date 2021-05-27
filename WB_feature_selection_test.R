source("WB_feature_selection.R")

# data preparation --------------------------------------------------------
load("data/artificial.rds")
colnames(artif_train_labels) <- "y"
artif <- cbind(artif_train, y=as.factor(artif_train_labels$y))


# Bagging feature importance
# slow
vi_bagging <- var_imp_bagging(artif)

# Random Forest feature importance

vi_rf_acc <- var_imp_rf_acc(artif)
vi_rf_gini <- var_imp_rf_gini(artif)

# chi square test
vi_chisq <- var_imp_chisq(artif)

# entropy-based
vi_inf_gain <- var_imp_inf_gain(artif)
vi_gain_rat <- var_imp_gain_rat(artif, biggest_diff = FALSE)
vi_sym_unc <- var_imp_sym_unc(artif)

# Recursive feature elimination
vi_rfe <- var_imp_rfe(artif)

# AIC bIC

#slow
vi_aic <- var_imp_AIC(artif)
vi_bic <- var_imp_BIC(artif)


# MCFS
# doesn't work
# somehow changes working directory
# actually works but not on whole dataset
vi_mcfs1 <- var_imp_mcfs(artif[,c(1:400, 501)])
vi_mcfs2 <- var_imp_mcfs(artif[,c(401:500, 501)])

vi_mcfs <- c(vi_mcfs1, vi_mcfs2)
