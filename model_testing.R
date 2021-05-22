load("data/artificial.rds")
colnames(artif_train_labels) <- "y"
artif <- cbind(artif_train, y=as.factor(artif_train_labels$y))

source('test_models.R')

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

