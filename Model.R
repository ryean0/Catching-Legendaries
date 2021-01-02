library(tidyverse)
library(dslabs)
library(tidytext)
library(dplyr)
library(caret)
library(matrixStats)
library(data.table)
library(stringr)
library(lubridate)
library(RColorBrewer)

#Start by loading the Pokemon Dataset

pokemon <- read.csv('C:/Users/User/Documents/Pokemon.csv')

#To ease prediction, convert the 'Legendary' column from Booleans into factors

pokemon$Legendary <- factor(pokemon$Legendary, levels = c('True', 'False'), labels = c(1,0))

# In order to train a good model, we first do a train-test split on the entire dataset and preserve our test set. Create a smaller training set to serve as a pseudo test set.

set.seed(1)
test_index <- createDataPartition(y = pokemon$Legendary, times = 1, p = 0.1, list = FALSE)
train_set <- pokemon[-test_index,]
test_set <- pokemon[test_index,]

train_index <- createDataPartition(y = train_set$Legendary, times = 1, p = 0.1, list = FALSE)
train_a <- train_set[-train_index,]
train_b <- train_set[train_index,]

# For all our visualizations after this point, we will only make use of the training set.

# We will first analyze the relationship between the total stat points of a pokemon and their legendary status

stat_effect <- train_set %>% ggplot(aes(x=Total, fill=Legendary)) + geom_histogram()

# From this, we can see that the Legendary pokemon start appearing around total statpoints=500+, and dominate the extreme right end of the statpoints
# I hypothesise that legendary pokemon might have much higher special attack and defense stats as compared to normal pokemon

sp_effect <- train_set %>% mutate(sp_stat = round(Sp..Atk+Sp..Def)) %>% ggplot() + geom_density(aes(x=sp_stat, fill=Legendary, alpha=0.5))

# From this we can see that special stats are a statistically significant predictor for legendary pokemon
# Next, we can attempt to see if the generation that the pokemon belong to actually play a role in determining their legendary status

gen_effect <- train_set %>% ggplot(aes(x=Generation, fill=Legendary)) + geom_bar()

# From this, we can see that gen 3 to gen 5 inclusive have a marked higher number of legendary pokemons

type_effect <- train_set %>% group_by(Type.1) %>% ggplot(aes(x=Type.1, fill=Legendary)) + geom_bar() + theme(axis.text.x=element_text(angle=90))

# Here we can see that some types do not have any legendaries while types like dragon and psychic have significantly more

#We will now create our ensemble using the 4 machine-learning algorithms, KNN, QDA, LDA and Random Forests. 

model_func <- function(test_set){

knn_control <- trainControl(method='cv', number=3, p=0.9)
knn_fit <- train(Legendary ~ HP + Attack + Defense + Sp..Atk + Sp..Def + Speed + Total + Generation, method='knn', 
                 tuneGrid=data.frame(k=seq(1,15,2)), trControl=knn_control, data=train_a)
knn_y_hat <- predict(knn_fit, newdata=test_set, type='raw') 
knn_y_hat <- ifelse(as.integer(knn_y_hat)==2, 0, 1)


qda_fit <- train(Legendary ~ HP + Attack + Defense + Sp..Atk + Sp..Def + Speed + Generation, method = 'qda', data=train_a)
qda_y_hat <- predict(qda_fit, test_set, type='raw')
qda_y_hat <- ifelse(as.integer(qda_y_hat)==2, 0, 1)

lda_fit <- train(Legendary ~ HP + Attack + Defense + Sp..Atk + Sp..Def + Speed + Generation, method = 'lda', data=train_a)
lda_y_hat <- predict(lda_fit, test_set, type='raw')
lda_y_hat <- ifelse(as.integer(lda_y_hat)==2, 0, 1)

rf_control <- trainControl(method='cv', number=3, p=0.9)
rf_grid <- expand.grid(minNode=c(1,5), predFixed=c(5,5,5,5,5))
rf_fit <- train(Legendary ~ HP + Attack + Defense + Sp..Atk + Sp..Def + Speed + Generation, 
                method='Rborist', ntree=50, trControl=rf_control, tuneGrid=rf_grid, nSamp=5000, data=train_a)
rf_y_hat <- predict(rf_fit, test_set)
rf_y_hat <- ifelse(as.integer(rf_y_hat)==2, 0, 1)

loess_fit <- train(Legendary ~ HP + Attack + Defense + Sp..Atk + Sp..Def + Speed + Generation, 
                   method = 'gamLoess', tuneGrid=expand.grid(span=seq(0.2, 0.6, len=2), degree=1), data=train_a)
loess_y_hat <- predict(loess_fit, test_set)
loess_y_hat <- ifelse(as.integer(loess_y_hat)==2, 0, 1)


final_y_hat <- (knn_y_hat+qda_y_hat+lda_y_hat+rf_y_hat+loess_y_hat)/4
final_y_hat <- ifelse(final_y_hat < 0.50, 0, 1)
final_y_hat <- final_y_hat %>% factor(levels=levels(test_set$Legendary))
return(confusionMatrix(data=final_y_hat, reference=test_set$Legendary)$overall['Accuracy'])

}


#Final test 

model_func(train_b)
model_func(test_set=test_set)


















