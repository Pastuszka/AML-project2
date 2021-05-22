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
test_nnet(subset_artif)

# digits ------------------------------------------------------------------

load("data/digits.rds")
colnames(digits_train_labels) <- "y"
digits <- cbind(digits_train, y=as.factor(digits_train_labels$y))

imp_columns <- get_important_columns(digits)

subset_digits <- digits[, imp_columns$selected_cols]