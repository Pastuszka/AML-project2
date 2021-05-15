library(funModeling)


digits_train <- read.table("data/digits_train.data")
digits_train_labels <- read.table("data/digits_train.labels")

head(digits_train)
status_dig <- status(digits_train)
View(status_dig)
# remove columns with only one unique value
one_unique <- which(status_dig$unique==1)
digits_train <- digits_train[,-one_unique]
save(digits_train, digits_train_labels, file = "data/digits.rds")
load("data/digits.rds")

artif_train <- read.table("data/artificial_train.data")
artif_train_labels <- read.table("data/artificial_train.labels")

status_art <- status(artif_train)
View(status_art)
which(status_art$unique==1)

save(artif_train, artif_train_labels, file = "data/artificial.rds")
