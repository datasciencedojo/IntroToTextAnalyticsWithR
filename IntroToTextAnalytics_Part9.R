#
# Copyright 2017 Data Science Dojo
#    
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 


#
# This R source code file corresponds to video 9 of the Data Science
# Dojo YouTube series "Introduction to Text Analytics with R" located 
# at the following URL:
#     https://www.youtube.com/watch?v=SgrLE6WQzkE     
#


# Install all required packages.
install.packages(c("ggplot2", "e1071", "caret", "quanteda", 
                   "irlba", "randomForest")) 




# Load up the .CSV data and explore in RStudio.
spam.raw <- read.csv("spam.csv", stringsAsFactors = FALSE, fileEncoding = "UTF-16")
View(spam.raw)


# Clean up the data frame and view our handiwork.
spam.raw <- spam.raw[, 1:2]
names(spam.raw) <- c("Label", "Text")
View(spam.raw)



# Check data to see if there are missing values.
length(which(!complete.cases(spam.raw)))



# Convert our class label into a factor.
spam.raw$Label <- as.factor(spam.raw$Label)



# The first step, as always, is to explore the data.
# First, let's take a look at distibution of the class labels (i.e., ham vs. spam).
prop.table(table(spam.raw$Label))



# Next up, let's get a feel for the distribution of text lengths of the SMS 
# messages by adding a new feature for the length of each message.
spam.raw$TextLength <- nchar(spam.raw$Text)
summary(spam.raw$TextLength)



# Visualize distribution with ggplot2, adding segmentation for ham/spam.
library(ggplot2)

ggplot(spam.raw, aes(x = TextLength, fill = Label)) +
  theme_bw() +
  geom_histogram(binwidth = 5) +
  labs(y = "Text Count", x = "Length of Text",
       title = "Distribution of Text Lengths with Class Labels")



# At a minimum we need to split our data into a training set and a
# test set. In a true project we would want to use a three-way split 
# of training, validation, and test.
#
# As we know that our data has non-trivial class imbalance, we'll 
# use the mighty caret package to create a randomg train/test split 
# that ensures the correct ham/spam class label proportions (i.e., 
# we'll use caret for a random stratified split).
library(caret)
help(package = "caret")


# Use caret to create a 70%/30% stratified split. Set the random
# seed for reproducibility.
set.seed(32984)
indexes <- createDataPartition(spam.raw$Label, times = 1,
                               p = 0.7, list = FALSE)

train <- spam.raw[indexes,]
test <- spam.raw[-indexes,]


# Verify proportions.
prop.table(table(train$Label))
prop.table(table(test$Label))



# Text analytics requires a lot of data exploration, data pre-processing
# and data wrangling. Let's explore some examples.

# HTML-escaped ampersand character.
train$Text[21]


# HTML-escaped '<' and '>' characters. Also note that Mallika Sherawat
# is an actual person, but we will ignore the implications of this for
# this introductory tutorial.
train$Text[38]


# A URL.
train$Text[357]



# There are many packages in the R ecosystem for performing text
# analytics. One of the newer packages in quanteda. The quanteda
# package has many useful functions for quickly and easily working
# with text data.
library(quanteda)
help(package = "quanteda")


# Tokenize SMS text messages.
train.tokens <- tokens(train$Text, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)

# Take a look at a specific SMS message and see how it transforms.
train.tokens[[357]]


# Lower case the tokens.
train.tokens <- tokens_tolower(train.tokens)
train.tokens[[357]]


# Use quanteda's built-in stopword list for English.
# NOTE - You should always inspect stopword lists for applicability to
#        your problem/domain.
train.tokens <- tokens_select(train.tokens, stopwords(), 
                              selection = "remove")
train.tokens[[357]]


# Perform stemming on the tokens.
train.tokens <- tokens_wordstem(train.tokens, language = "english")
train.tokens[[357]]


# Create our first bag-of-words model.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)


# Transform to a matrix and inspect.
train.tokens.matrix <- as.matrix(train.tokens.dfm)
View(train.tokens.matrix[1:20, 1:100])
dim(train.tokens.matrix)


# Investigate the effects of stemming.
colnames(train.tokens.matrix)[1:50]


# Per best practices, we will leverage cross validation (CV) as
# the basis of our modeling process. Using CV we can create 
# estimates of how well our model will do in Production on new,
# unseen data. CV is powerful, but the downside is that it
# requires more processing and therefore more time.
#
# If you are not familiar with CV, consult the following 
# Wikipedia article:
#
#   https://en.wikipedia.org/wiki/Cross-validation_(statistics)
#

# Setup a the feature data frame with labels.
train.tokens.df <- cbind(Label = train$Label, data.frame(train.tokens.dfm))


# Often, tokenization requires some additional pre-processing
names(train.tokens.df)[c(146, 148, 235, 238)]


# Cleanup column names.
names(train.tokens.df) <- make.names(names(train.tokens.df))


# Use caret to create stratified folds for 10-fold cross validation repeated 
# 3 times (i.e., create 30 random stratified samples)
set.seed(48743)
cv.folds <- createMultiFolds(train$Label, k = 10, times = 3)

cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)


# Our data frame is non-trivial in size. As such, CV runs will take 
# quite a long time to run. To cut down on total execution time, use
# the doSNOW package to allow for multi-core training in parallel.
#
# WARNING - The following code is configured to run on a workstation-
#           or server-class machine (i.e., 12 logical cores). Alter
#           code to suit your HW environment.
#
#install.packages("doSNOW")
library(doSNOW)


# Time the code execution
start.time <- Sys.time()


# Create a cluster to work on 10 logical cores.
cl <- makeCluster(10, type = "SOCK")
registerDoSNOW(cl)


# As our data is non-trivial in size at this point, use a single decision
# tree alogrithm as our first model. We will graduate to using more 
# powerful algorithms later when we perform feature extraction to shrink
# the size of our data.
rpart.cv.1 <- train(Label ~ ., data = train.tokens.df, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)


# Processing is done, stop cluster.
stopCluster(cl)


# Total time of execution on workstation was approximately 4 minutes. 
total.time <- Sys.time() - start.time
total.time


# Check out our results.
rpart.cv.1



# The use of Term Frequency-Inverse Document Frequency (TF-IDF) is a 
# powerful technique for enhancing the information/signal contained
# within our document-frequency matrix. Specifically, the mathematics
# behind TF-IDF accomplish the following goals:
#    1 - The TF calculation accounts for the fact that longer 
#        documents will have higher individual term counts. Applying
#        TF normalizes all documents in the corpus to be length 
#        independent.
#    2 - The IDF calculation accounts for the frequency of term
#        appearance in all documents in the corpus. The intuition 
#        being that a term that appears in every document has no
#        predictive power.
#    3 - The multiplication of TF by IDF for each cell in the matrix
#        allows for weighting of #1 and #2 for each cell in the matrix.


# Our function for calculating relative term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}

# Our function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))

  log10(corpus.size / doc.count)
}

# Our function for calculating TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}


# First step, normalize all documents via TF.
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)
View(train.tokens.df[1:20, 1:100])


# Second step, calculate the IDF vector that we will use - both
# for training data and for test data!
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)


# Lastly, calculate TF-IDF for our training corpus.
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])


# Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])


# Check for incopmlete cases.
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases]


# Fix incomplete cases
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))
dim(train.tokens.tfidf)
sum(which(!complete.cases(train.tokens.tfidf)))


# Make a clean data frame using the same process as before.
train.tokens.tfidf.df <- cbind(Label = train$Label, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))


# Time the code execution
start.time <- Sys.time()

# Create a cluster to work on 10 logical cores.
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# As our data is non-trivial in size at this point, use a single decision
# tree alogrithm as our first model. We will graduate to using more 
# powerful algorithms later when we perform feature extraction to shrink
# the size of our data.
rpart.cv.2 <- train(Label ~ ., data = train.tokens.tfidf.df, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time

# Check out our results.
rpart.cv.2



# N-grams allow us to augment our document-term frequency matrices with
# word ordering. This often leads to increased performance (e.g., accuracy)
# for machine learning models trained with more than just unigrams (i.e.,
# single terms). Let's add bigrams to our training data and the TF-IDF 
# transform the expanded featre matrix to see if accuracy improves.

# Add bigrams to our feature matrix.
train.tokens <- tokens_ngrams(train.tokens, n = 1:2)
train.tokens[[357]]


# Transform to dfm and then a matrix.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)
train.tokens.matrix <- as.matrix(train.tokens.dfm)
train.tokens.dfm


# Normalize all documents via TF.
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)

  
# Calculate the IDF vector that we will use for training and test data!
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)

  
# Calculate TF-IDF for our training corpus 
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, 
                             idf = train.tokens.idf)

  
# Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)

  
# Fix incomplete cases
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))


# Make a clean data frame.
train.tokens.tfidf.df <- cbind(Label = train$Label, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))


# Clean up unused objects in memory.
gc()




#
# NOTE - The following code requires the use of command-line R to execute
#        due to the large number of features (i.e., columns) in the matrix.
#        Please consult the following link for more details if you wish
#        to run the code yourself:
#
#        https://stackoverflow.com/questions/28728774/how-to-set-max-ppsize-in-r
#
#        Also note that running the following code required approximately
#        38GB of RAM and more than 4.5 hours to execute on a 10-core 
#        workstation!
#


# Time the code execution
# start.time <- Sys.time()

# Leverage single decision trees to evaluate if adding bigrams improves the 
# the effectiveness of the model.
# rpart.cv.3 <- train(Label ~ ., data = train.tokens.tfidf.df, method = "rpart", 
#                     trControl = cv.cntrl, tuneLength = 7)

# Total time of execution on workstation was
# total.time <- Sys.time() - start.time
# total.time

# Check out our results.
# rpart.cv.3

#
# The results of the above processing show a slight decline in rpart 
# effectiveness with a 10-fold CV repeated 3 times accuracy of 0.9457.
# As we will discuss later, while the addition of bigrams appears to 
# negatively impact a single decision tree, it helps with the mighty
# random forest!
#




# We'll leverage the irlba package for our singular value 
# decomposition (SVD). The irlba package allows us to specify
# the number of the most important singular vectors we wish to
# calculate and retain for features.
library(irlba)


# Time the code execution
start.time <- Sys.time()

# Perform SVD. Specifically, reduce dimensionality down to 300 columns
# for our latent semantic analysis (LSA).
train.irlba <- irlba(t(train.tokens.tfidf), nv = 300, maxit = 600)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time


# Take a look at the new feature data up close.
View(train.irlba$v)


# As with TF-IDF, we will need to project new data (e.g., the test data)
# into the SVD semantic space. The following code illustrates how to do
# this using a row of the training data that has already been transformed
# by TF-IDF, per the mathematics illustrated in the slides.
#
#
sigma.inverse <- 1 / train.irlba$d
u.transpose <- t(train.irlba$u)
document <- train.tokens.tfidf[1,]
document.hat <- sigma.inverse * u.transpose %*% document

# Look at the first 10 components of projected document and the corresponding
# row in our document semantic space (i.e., the V matrix)
document.hat[1:10]
train.irlba$v[1, 1:10]



#
# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).
#
train.svd <- data.frame(Label = train$Label, train.irlba$v)


# Create a cluster to work on 10 logical cores.
cl <- makeCluster(10, type = "SOCK")
registerDoSNOW(cl)

# Time the code execution
start.time <- Sys.time()

# This will be the last run using single decision trees. With a much smaller
# feature matrix we can now use more powerful methods like the mighty Random
# Forest from now on!
rpart.cv.4 <- train(Label ~ ., data = train.svd, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time

# Check out our results.
rpart.cv.4




#
# NOTE - The following code takes a long time to run. Here's the math.
#        We are performing 10-fold CV repeated 3 times. That means we
#        need to build 30 models. We are also asking caret to try 7 
#        different values of the mtry parameter. Next up by default
#        a mighty random forest leverages 500 trees. Lastly, caret will
#        build 1 final model at the end of the process with the best 
#        mtry value over all the training data. Here's the number of 
#        tree we're building:
#
#             (10 * 3 * 7 * 500) + 500 = 105,500 trees!
#
# On a workstation using 10 cores the following code took 28 minutes 
# to execute.
#


# Create a cluster to work on 10 logical cores.
# cl <- makeCluster(10, type = "SOCK")
# registerDoSNOW(cl)

# Time the code execution
# start.time <- Sys.time()

# We have reduced the dimensionality of our data using SVD. Also, the 
# application of SVD allows us to use LSA to simultaneously increase the
# information density of each feature. To prove this out, leverage a 
# mighty Random Forest with the default of 500 trees. We'll also ask
# caret to try 7 different values of mtry to find the mtry value that 
# gives the best result!
# rf.cv.1 <- train(Label ~ ., data = train.svd, method = "rf", 
#                 trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
# stopCluster(cl)

# Total time of execution on workstation was 
# total.time <- Sys.time() - start.time
# total.time


# Load processing results from disk!
load("rf.cv.1.RData")

# Check out our results.
rf.cv.1

# Let's drill-down on the results.
confusionMatrix(train.svd$Label, rf.cv.1$finalModel$predicted)





# OK, now let's add in the feature we engineered previously for SMS 
# text length to see if it improves things.
train.svd$TextLength <- train$TextLength


# Create a cluster to work on 10 logical cores.
# cl <- makeCluster(10, type = "SOCK")
# registerDoSNOW(cl)

# Time the code execution
# start.time <- Sys.time()

# Re-run the training process with the additional feature.
# rf.cv.2 <- train(Label ~ ., data = train.svd, method = "rf",
#                 trControl = cv.cntrl, tuneLength = 7, 
#                 importance = TRUE)

# Processing is done, stop cluster.
# stopCluster(cl)

# Total time of execution on workstation was 
# total.time <- Sys.time() - start.time
# total.time

# Load results from disk.
load("rf.cv.2.RData")

# Check the results.
rf.cv.2

# Drill-down on the results.
confusionMatrix(train.svd$Label, rf.cv.2$finalModel$predicted)

# How important was the new feature?
library(randomForest)
varImpPlot(rf.cv.1$finalModel)
varImpPlot(rf.cv.2$finalModel)

