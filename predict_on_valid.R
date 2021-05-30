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

learner <- lrn("classif.kknn", kernel='biweight', k=18, predict_type='prob')
full_learner <- learner$train(task_artif)


artif_valid <- read.table("data/artificial_valid.data")

artif_valid_selected_columns <- c("V242", "V339", "V106", "V49", "V337",
                                       "V473", "V282", "V454", "V29")
artif_valid <- artif_valid[,artif_valid_selected_columns]

prediction <- predict(learner, artif_valid, predict_type='prob')

aritf_prediction <- data.frame('WOJBOG'=prediction[,2])

write.csv(aritf_prediction, file='WOJBOG_artificial_prediction.txt', row.names = FALSE)

artif_columns <- data.frame('WOJBOG'= c(242, 339, 106, 49, 337,
                                       473, 282, 454, 29))
write.csv(artif_columns, file='WOJBOG_artificial_features.txt', row.names = FALSE)


#### digits ####

load("data/digits.rds")
colnames(digits_train_labels) <- "y"
digits <- cbind(digits_train, y=as.factor(digits_train_labels$y))

measure <- msr("classif.bacc")

digits_selected <- c("V3657", "V3976", "V558",  "V1229", "V3003", "V4508",
                     "V512",  "V1213", "V140",  "V1505", "V1643", "V2008",
                     "V215",  "V2489", "V2743", "V2813", "V3063", "V3066",
                     "V3172", "V3305", "V3328", "V3361", "V3469", "V3544",
                     "V3722", "V4107", "V4229", "V4268", "V4387", "V4410",
                     "V4425", "V4554", "V4576", "V4586", "V4964", "V555", 
                     "V569",  "V577",  "V683",  "V84",   "V881",  "V949")
digits_selected_withy <- c(digits_selected, 'y')


digits_subset <- digits[,digits_selected_withy]

task_digits <- TaskClassif$new(id = "digits", 
                              backend = digits_subset, 
                              target = "y")
task_digits

learner <- lrn("classif.svm", kernel='radial', cost=5, 
               type = 'C-classification', predict_type='prob')

train_set <- sample(task_digits$nrow, 0.9 * task_digits$nrow)
test_set <- setdiff(seq_len(task_digits$nrow), train_set)

learner$train(task_digits, row_ids = train_set)

prediction <- learner$predict(task_digits, row_ids = test_set)

prediction$score(measure)

learner <- lrn("classif.svm", kernel='radial', cost=5,
               type = 'C-classification', predict_type='prob')
full_learner <- learner$train(task_digits)


digits_valid <- read.table("data/digits_valid.data")

digits_valid <- digits_valid[,digits_selected]

prediction <- predict(learner, digits_valid, predict_type='prob')

digits_prediction <- data.frame('WOJBOG'=prediction[,2])

write.csv(digits_prediction, file='WOJBOG_digits_prediction.txt', row.names = FALSE)

digits_columns <- data.frame('WOJBOG'= c(3657, 3976, 558,  1229, 3003, 4508,
                                            512,  1213, 140,  1505, 1643, 2008,
                                            215,  2489, 2743, 2813, 3063, 3066,
                                            3172, 3305, 3328, 3361, 3469, 3544,
                                            3722, 4107, 4229, 4268, 4387, 4410,
                                            4425, 4554, 4576, 4586, 4964, 555, 
                                            569,  577,  683,  84,   881,  949))
write.csv(digits_columns, file='WOJBOG_digits_features.txt', row.names = FALSE)
