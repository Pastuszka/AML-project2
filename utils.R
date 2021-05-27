get_important_columns <- function(data){
  source("WB_feature_selection.R")
  source("MP_feature_selection.R")
  n <- ncol(data)
  # print('vi_bagging')
  # vi_bagging <- var_imp_bagging(data)
  print('vi_ran_imp')
  vi_ran_imp <- var_imp_ranger_impurity(data)
  print('vi_ran_perm')
  vi_ran_perm <- var_imp_ranger_perm(data)
  print('vi_chisq')
  vi_chisq <- var_imp_chisq(data)
  print('vi_inf_gain')
  vi_inf_gain <- var_imp_inf_gain(data)
  print('vi_gain_rat')
  vi_gain_rat <- var_imp_gain_rat(data, biggest_diff = FALSE)
  print('vi_sym_unc')
  vi_sym_unc <- var_imp_sym_unc(data)
  print('vi_rfe')
  vi_rfe <- var_imp_rfe(data)
  # print('vi_bic')
  # vi_bic <- var_imp_BIC(data)
  # print('vi_mcfs')
  # vi_mcfs <- c()
  # if (n==501){
  #   for (i in 1:2){
  #     vi_mcfs <- c(vi_mcfs, var_imp_mcfs(data[,c((250*(i-1)+1):250*(i), n)]))
  #   }
  # }
  # else{
  #   for (i in 1:floor(n/400)){
  #     vi_mcfs <- c(vi_mcfs, var_imp_mcfs(data[,c((400*(i-1)+1):400*(i), n)]))
  #   }
  # }
  # vi_mcfs <- c(vi_mcfs, var_imp_mcfs(data[,c((400*(i)+1):(n-1), n)]))
  print('vi_lasso')
  vi_lasso <- var_imp_lasso(data)
  print('vi_lasso_1se')
  vi_lasso_1se <- var_imp_lasso(data, TRUE)
  print('vi_caret')
  vi_caret <- var_imp_caret(data)
  print('vi_corr')
  vi_corr <- var_imp_corr(data)
  print('vi_boruta')
  vi_boruta <- var_imp_boruta(data)
  vi_columns <- list(vi_boruta, 
                  vi_corr,
                  vi_caret,
                  vi_chisq,
                  vi_gain_rat, 
                  vi_inf_gain, 
                  vi_lasso, 
                  vi_lasso_1se, 
                  vi_ran_imp, 
                  vi_ran_perm, 
                  vi_rfe, 
                  vi_sym_unc)
  names(vi_columns) <- c('vi_boruta', 
                         'vi_corr',
                         'vi_caret',
                         'vi_chisq',
                         'vi_gain_rat', 
                         'vi_inf_gain', 
                         'vi_lasso', 
                         'vi_lasso_1se', 
                         'vi_ran_imp', 
                         'vi_ran_perm', 
                         'vi_rfe', 
                         'vi_sym_unc')
  columns_freq <- sort(table(unlist(vi_columns)), decreasing=TRUE)
  return(structure(list(all_results=vi_columns, selected_cols=names(columns_freq)[which(columns_freq>2)])))
}

