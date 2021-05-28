library(mlr3verse)

load("data/artificial.rds")
colnames(artif_train_labels) <- "y"
artif <- cbind(artif_train, y=as.factor(artif_train_labels$y))

measure <- msr("classif.bacc")

artif_ranger_decol <- c("V242", "V339", "V106", "V49", "V337", "V473", "V282",
                        "V454", "V29", 'y')

artif_decol <- artif[,artif_ranger_decol]

task_artif <- TaskClassif$new(id = "artif", 
                              backend = artif_decol, 
                              target = "y")
task_artif

learner <- lrn("classif.kknn", kernel='biweight', k=18, predict_type='prob')

train_set <- sample(task_artif$nrow, 0.9 * task_artif$nrow)
test_set <- setdiff(seq_len(task_artif$nrow), train_set)

learner$train(task_artif, row_ids = train_set)

prediction <- learner$predict(task_artif, row_ids = test_set)

prediction$score(measure)


full_learner <- learner$train(task_artif)


digits_valid <- read.table("data/digits_valid.data")

digits_valid_selected_columns <- c("V242", "V339", "V106", "V49", "V337",
                                       "V473", "V282", "V454", "V29")
digits_valid <- digits_valid[,digits_valid_selected_columns]

prediction <- predict(learner, digits_valid, predict_type='prob')

aritf_prediction <- data.frame('WOJBOG'=prediction[,2])

write.csv(aritf_prediction, file='WOJBOG_artificial_prediction.txt', row.names = FALSE)

artif_columns <- data.frame('WOJBOG'= c(242, 339, 106, 49, 337,
                                       473, 282, 454, 29))
write.csv(artif_columns, file='WOJBOG_artificial_features.txt', row.names = FALSE)
