get_important_columns <- function(data){
  source("WB_feature_selection.R")
  source("MP_feature_selection.R")
  vi_bagging <- var_imp_bagging(data)
  vi_rf_acc <- var_imp_rf_acc(data)
  vi_rf_gini <- var_imp_rf_gini(data)
  vi_chisq <- var_imp_chisq(data)
  vi_inf_gain <- var_imp_inf_gain(data)
  vi_gain_rat <- var_imp_gain_rat(data, biggest_diff = FALSE)
  vi_sym_unc <- var_imp_sym_unc(data)
  vi_rfe <- var_imp_rfe(data)
  vi_bic <- var_imp_BIC(data)
  vi_mcfs <- c()
  if (cnol(data)==501){
    for (i in 1:2){
      vi_mcfs <- c(vi_mcfs, var_imp_mcfs(data[,c((250*(i-1)+1):250*(i-1), 501)]))
    }
  }
  else{
    for (i in 1:5){
      vi_mcfs <- c(vi_mcfs, var_imp_mcfs(data[,c((400*(i-1)+1):400*(i-1), 2001)]))
    }
  }
  vi_lasso <- var_imp_lasso(data)
  vi_lasso_1se <- var_imp_lasso(data, TRUE)
  vi_caret <- var_imp_caret(data)
  vi_corr <- var_imp_corr(data)
  vi_boruta <- var_imp_boruta(data)
  vi_columns <- list(vi_bagging, 
                  vi_boruta, 
                  vi_bic, 
                  vi_corr,
                  vi_caret,
                  vi_chisq,
                  vi_gain_rat, 
                  vi_inf_gain, 
                  vi_lasso, 
                  vi_lasso_1se, 
                  vi_mcfs, 
                  vi_rf_acc, 
                  vi_rf_gini, 
                  vi_rfe, 
                  vi_sym_unc)
  names(vi_columns) <- c('vi_bagging', 
                         'vi_boruta', 
                         'vi_bic', 
                         'vi_corr',
                         'vi_caret',
                         'vi_gain_rat', 
                         'vi_inf_gain', 
                         'vi_lasso', 
                         'vi_lasso_1se', 
                         'vi_mcfs', 
                         'vi_rf_acc', 
                         'vi_rf_gini', 
                         'vi_rfe', 
                         'vi_sym_unc')
  columns_freq <- sort(table(unlist(vi_columns)), decreasing=TRUE)
  return(structure(list(all_results=vi_columns, selected_cols=names(columns_freq)[which(columns_freq>2)])))
}

