library(readr)
library(tidyverse)
library(tidyr)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(MASS)
library(reshape2)
library(Hmisc)
library(funModeling)
library(corrplot)
library(ggcorrplot)
library(xgboost)
library(e1071)
library(caret)
library(DiagrammeR)
library(ROCR)
library(dplyr)
library(car)
library(magrittr)
# Kickstarter Projects dataset

# Dataset available from: https://www.kaggle.com/kemical/kickstarter-projects

###########################
###     Code Starts     ###
###########################

# Set working directory
getwd()
setwd("C:/Users/Alexl/Documents/Projects")

# Import the data
raw_dataset2018 <- as.data.frame(read.csv("ks-projects-201801.csv", header = TRUE))
# View(raw_dataset2018)

# understand the dataset - data mining
names(raw_dataset2018)
str(raw_dataset2018)
glimpse(raw_dataset2018)
summary(raw_dataset2018)

# convert these variables to factors using - as factors....
# category
# main_category
# currency
# state
# country

raw_dataset2018$category <- as.factor(raw_dataset2018$category)
raw_dataset2018$main_category <- as.factor(raw_dataset2018$main_category)
raw_dataset2018$currency <- as.factor(raw_dataset2018$currency)
raw_dataset2018$state <- as.factor(raw_dataset2018$state)
raw_dataset2018$country <- as.factor(raw_dataset2018$country)

levels(raw_dataset2018$state)


tidy.2018 <- raw_dataset2018

describe(tidy.2018)


####################################################
###       Viszulize Categorical variables        ###
####################################################
# freqeuncy tables of new categorical variables

# States
StateFreqTable <- tidy.2018 %>%
  group_by(state)%>%
  summarise(counts = n())
StateFreqTable

ggplot(StateFreqTable, aes(x = state, y = counts)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = counts), vjust = -0.3) + 
  theme_pubclean() +
  ggtitle("Frequency of different project states") + 
  geom_jitter(width=0.15) +
  theme(axis.text.x = element_text(angle = 45, hjust=1))


# main_category
main_categoryFreqTable <- tidy.2018 %>%
  group_by(main_category)%>%
  summarise(counts = n())
main_categoryFreqTable

ggplot(main_categoryFreqTable, aes(x = main_category, y = counts)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = counts), vjust = -0.3) + 
  theme_pubclean() +
  ggtitle("Frequency of different main_category") +
  geom_jitter(width=0.15) +
  theme(axis.text.x = element_text(angle = 45, hjust=1))

# currency
currencyFreqTable <- tidy.2018 %>%
  group_by(currency)%>%
  summarise(counts = n())
currencyFreqTable

ggplot(currencyFreqTable, aes(x = currency, y = counts)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = counts), vjust = -0.3) + 
  theme_pubclean() +
  ggtitle("Frequency of different currency") +
  geom_jitter(width=0.15) +
  theme(axis.text.x = element_text(angle = 45, hjust=1))

# country
countryFreqTable <- tidy.2018 %>%
  group_by(country)%>%
  summarise(counts = n())
countryFreqTable

ggplot(countryFreqTable, aes(x = country, y = counts)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = counts), vjust = -0.3) + 
  theme_pubclean() +
  ggtitle("Frequency of different country") +
  geom_jitter(width=0.15) +
  theme(axis.text.x = element_text(angle = 45, hjust=1))


###################################
###     Numerical Variables     ###
###################################
numerical.2018 <- select_if(tidy.2018, is.numeric)
names(numerical.2018)
glimpse(numerical.2018)
describe(numerical.2018)
summary(numerical.2018)
class(numerical.2018)
head(numerical.2018)

summarise(numerical.2018$goal)

numerical.2018 <- na.omit(numerical.2018) # Remove NAs
numerical.2018[is.na(numerical.2018)] <- 0 # convert all NAs into zeros, no meaningful Zeros

#_________________________________________
# calculating the approprate bin width for histograms
# bw <- 2 * IQR(numerical.2018$) / length(numerical.2018$)^(1/3) 
# Freedman Diaconis rule for calculating bins
#__________________________________________

#Plot of goal amounts




#############################
###     Correlation       ###
#############################

# Corr Plot prep
corr <-  round(cor(numerical.2018))
p.mat <- cor_pmat(numerical.2018)

summary(p.mat)
# Get the upper triangle viz.,
ggcorrplot(corr, 
           hc.order = TRUE, 
           type = "lower",
           p.mat = p.mat, 
           insig = "blank")

# to get the correlation analyses results in a dataframe
corr_DF <-  ggstatsplot::ggcorrmat(
  data = numerical.2018,
  output = "dataframe"
) filter(state=="successful")
successful_df
glimpse(corr_DF)


################################
###     XG BOOST  Model     ####
###############################
#     Looking to predict which leads to a successful state of kickstarter projects.

### Prep dataframe for XG Boost
names(tidy.2018)
d1 <- tidy.2018
levels(d1$state)
head(d1)
names(d1)
# state Failed as a logic question, either true or false 1,0
successful_df <- d1 %>%
  filter(state=="successful")
 
names(successful_df)
#seperate to a list of names only
successful_df %>% 
  dplyr:::select("name")

dim(successful_df)
dim(d1)
#re_exports$CoffeeBeltCat<- ifelse(re_exports$re.exports %in% List_Coffeebelt, 1, 0)
d1$successful <- as.factor(ifelse(d1$state == "successful", 1, 0))
glimpse(d1)
summary(d1$successful)
head(d1)
View(d1)
#select independent variables for XG boost model
dxg <- d1%>%
  dplyr:::select( 'main_category', 
                  'goal', 
                  'pledged', 
                  'backers', 
                  'successful')  

glimpse(dxg)

dxg$successful <- as.factor(dxg$successful)
dxg$goal <- as.integer(dxg$goal)
glimpse(dxg)
summary(dxg)

dxg <- na.omit(dxg) # Remove NAs
dxg[is.na(dxg)] <- 0 # convert all NAs into zeros, no meaningful Zeros
View(dxg)
class(dxg$state)
dxg$state <- as.factor(dxg$state)
freq(dxg$state)
## 70% of the sample size
smp_size <- floor(0.70 * nrow(dxg))

## set the seed to make your partition reproducible
set.seed(223)
train_ind <- sample(seq_len(nrow(dxg)), size = smp_size)
train <- dxg[train_ind, ]
test <- dxg[-train_ind, ]
names(train)
names(test)

dim(train)
dim(test)

trainlabel <- as.numeric(as.factor(train$successful))-1 # set training label
testlabel <- as.numeric(as.factor(test$successful))-1   #set test Label

train$successful <- NULL   # remove from data
test$successful <- NULL

trainmat <- data.matrix(train)
testmat <- data.matrix(test)

##put our testing & training data into seperate Dmatrixs objects
dtrain <- xgb.DMatrix(data = trainmat, label= trainlabel)
dtest <- xgb.DMatrix(data = testmat, label= testlabel)

# Run XG boost model  on dtrained dataset aboce
xgmodel <- xgboost(data = dtrain, # the data   
                   nround = 50,
                   max.depth = 3,# boosting iterations
                   objective = "binary:logistic")  # the objective function
pred <- predict(xgmodel, dtest)
print(xgmodel)
summary(pred)
#error rate
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# plot the features
xgb.plot.multi.trees(feature_names = names(trainmat), 
                     model = xgmodel)
# convert log odds to probability
odds_to_probs <- function(odds){
  return(exp(odds)/ (1 + exp(odds)))
}
# probability of top leaf
odds_to_probs(1.1101)    #0.7521478

# get information on how important each feature is 
importance_matrix <- xgb.importance(names(trainmat), model = xgmodel) 
#plotting importance
xgb.plot.importance(importance_matrix)



#######################################
###     Optimization - XG BOOST     ###
#######################################

# get the number of negative & positive cases in our data
negative_cases <- sum(trainlabel == 0)
postive_cases <- sum(testlabel == 1)

model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 5, # the maximum depth of each decision tree
                       nround = 50, # number of boosting rounds
                       early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic", # the objective function
                       scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
                       gamma = 1) # add a regularization term

pred <- predict(model_tuned, dtest)
print(model_tuned)
summary(pred)
#error rate
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# plot the features
xgb.plot.multi.trees(feature_names = names(trainmat), 
                     model = model_tuned)
# convert log odds to probability
odds_to_probs <- function(odds){
  return(exp(odds)/ (1 + exp(odds)))
}
# probability of top leaf
odds_to_probs(1.8052)    #0.8587807

# get information on how important each feature is
importance_matrix <- xgb.importance(names(trainmat), model = model_tuned)
#plotting importance
xgb.plot.importance(importance_matrix)
