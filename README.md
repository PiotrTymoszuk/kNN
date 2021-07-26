# kNN
k-nearest neighbor prediction for binary outcome and various distance measures

A set of R functions used to generate kNN predictions of a any factorized outcome with various distance measures provided by distance() function from package philentropy and to test prediction accuracy (rates of correct predictions, error rates, sensitivity and specificity). Additional classifier/modeling functions: naive Bayes classification (provided by e1071 package), random trees (provided by randomForest), distance weighted kNN and accuracy testing for non-binary factor variable were implemented as well.

New: as random forests may model regression as well the accuracy assessing function accouts for that now and returns numeric measures of model error and fit.
