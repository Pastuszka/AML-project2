source("MP_feature_selection.R")
# data preparation --------------------------------------------------------
load("data/artificial.rds")
colnames(artif_train_labels) <- "y"
artif <- cbind(artif_train, y=as.factor(artif_train_labels$y))


# Lasso feature importance
vi_lasso <- var_imp_lasso(artif)
vi_lasso_1se <- var_imp_lasso(artif, TRUE)

# Caret feature importance
#slow
vi_caret <- var_imp_caret(artif)

#DALEX feature importance-srance
#slow but that's fine because it doesn't work anyway
vi_dalex <- var_imp_DALEX(artif) #broken dunskie warchlaki doesn't work

#Correlation feature importance
vi_corr <- var_imp_corr(artif)

#Boruta feature importance
#kinda slow
vi_boruta <- var_imp_boruta(artif)

#Genetic feature importance
#extremely slow, use at own risk
#no, I mean like reeeeeeeeeeaaaaaaaaalllllllllyyyyyyyyyyy sloooooooow
vi_genetic <- var_imp_genetic(artif, 100)
