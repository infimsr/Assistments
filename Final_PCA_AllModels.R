## Final_PCA_AllModels

rm(list=ls())

library(MASS)
library(caret)
library(rpart)
library(rpart.plot)
library(FNN)

## Importing the file
df <- read.csv("labelled_training_data.csv")
df$X <- NULL

## creating pivots
library(data.table)
dft <- as.data.table(df)
df_piv1 <- as.data.frame(dft[,.(isSTEM = max(isSTEM),
                                AveKnow = mean(AveKnow),
                                frattempts = max(totalFrAttempted),
                                problems = length(unique(problemId)),
                                hints = sum(hint),
                                HintPrbRatio = sum(hint)/length(unique(problemId)),
                                scaffolds = sum(scaffold),
                                ScafPrbRatio = sum(scaffold)/length(unique(problemId)),
                                originals = sum(original),
                                bottomhints = sum(bottomHint),
                                botPrbRatio = sum(bottomHint)/length(unique(problemId)),
                                frhelp = sum(frIsHelpRequest),
                                atschool = sum(frWorkingInSchool),
                                helpAccessUnder2 = sum(helpAccessUnder2Sec),
                                timeGreater10SecAndNextActionRight = sum(timeGreater10SecAndNextActionRight),
                                timeOver80 = sum(timeOver80),
                                AveCorrect.x = mean(AveCorrect.x)
),by=list(ITEST_id)])

# Data Input
pivots <- list()
first_pivot <- df_piv1
first_pivot$isSTEM <- as.factor(first_pivot$isSTEM)

pivots[[1]] <- first_pivot

# Applying pca
pcs <- prcomp(first_pivot[,-c(1:2)], scale. = T)
summary(pcs)
pcs$x

PoV <- pcs$sdev^2/sum(pcs$sdev^2)

i <- 0
sum <- 0
PrinComps <- 0
for (i in 1:length(PoV)) {
  sum <- sum + PoV[i]
  if (sum > 0.9) {
    PrinComps <- i
    break()
  }
}


pc_piv <- as.data.frame(pcs$x[,1:PrinComps])
pc_piv <- cbind(df_piv1[,1:2], pc_piv)

pivots[[2]] <- pc_piv

### Running all models and computing accuracies ###

model <- list()
accuracy <- vector()
i = 1
for (i in 1:length(pivots)) {
  j = 3*i
  #model exploration
  
  condensed <- pivots[[i]]
  
  
  # Splitting the condensed emo file into training and validation data set
  set.seed(123)
  training.index <- createDataPartition(condensed$ITEST_id, p = 0.6, list = FALSE)
  con.train <- condensed[training.index, ]
  con.valid <- condensed[-training.index, ]
  
  # Applying linear discriminant analysis on the training set
  lda.train <- lda(isSTEM~., data = con.train)
  pred.valid <- predict(lda.train, con.valid)
  model[[j]] <- lda.train
  
  
  # Confusion Matrix
  tab <- table(pred.valid$class, con.valid$isSTEM)
  LDAaccuracy <- sum(diag(tab))/sum(tab)
  accuracy[[j]] <- LDAaccuracy
  
  # Applying Logistic Regression on the training set
  logit.reg <- glm(isSTEM ~ ., data = con.train, family = "binomial") 
  options(scipen=999)
  logit.valid <- predict(logit.reg,con.valid)
  model[[j+1]] <- logit.reg
  
  # Confusion Matrix for Logistic Regression
  tab1 <- table(logit.valid>0, con.valid$isSTEM)
  LOGITaccuracy <- sum(diag(tab1))/sum(tab1)
  accuracy[[j+1]] <- LOGITaccuracy
  
  # Applying CART
  ct.train <- rpart(isSTEM ~ ., data = con.train, method = "class")
  ct.valid <- predict(ct.train, con.valid,type = "class")
  model[[j+2]] <- ct.train
  
  # Confusion Matrix for CART
  tab2 <- table(ct.valid, con.valid$isSTEM)
  CARTaccuracy <- sum(diag(tab2))/sum(tab2)
  accuracy[[j+2]] <- CARTaccuracy
  
}