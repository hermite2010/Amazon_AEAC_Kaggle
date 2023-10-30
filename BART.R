library(tidymodels)
library(embed)     # For target encoding
library(vroom) 
library(parallel)
#library(ggmosaic)  # For plotting
#library(ranger)    # FOR RANDOM/CLASSIFICATION FOREST
library(doParallel)
library(discrim) # FOR NAIVE BAYES
library(kknn)
library(parsnip)    # FOR BART
library(dbarts)
library(themis)

# Reading in the Data
AEAC_Train <- vroom("train.csv") #"Amazon_AEAC_Kaggle/train.csv" for local
AEAC_Test <- vroom("test.csv") #"Amazon_AEAC_Kaggle/test.csv" for local

AEAC_Train$ACTION = as.factor(AEAC_Train$ACTION)

AEAC_recipe <- recipe(ACTION ~., data=AEAC_Train) %>% 
  step_mutate_at(all_numeric_predictors(), fn= factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_bayes(all_nominal_predictors(), outcome= vars(ACTION)) %>% 
  step_smote(all_outcomes(), neighbors = 5) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = 0.95)
# Try step_lencode_bayes() in the future

## Set up Parallel processing
num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
if (num_cores > 4)
  num_cores = 10
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

prepped_recipe <- prep(AEAC_recipe)
baked_data <- bake(prepped_recipe, new_data=AEAC_Train)

# Set the model
bart_mod <- parsnip::bart(mode = "classification",
                          engine = "dbarts",
                          trees = 25)

# Set workflow
bart_wf <- workflow() %>%
  add_recipe(AEAC_recipe) %>%
  add_model(bart_mod) %>% 
  fit(data = AEAC_Train)

stopCluster(cl)

# Finalize and Predict 
bart_preds <- bart_wf %>%
  fit(data = AEAC_Train) %>% 
  predict(new_data = AEAC_Test, type="prob") %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_1) %>% 
  select(3,2)


vroom_write(x=bart_preds, file="Bart_pca+smote.csv", delim=",") #"Amazon_AEAC_Kaggle/NaiveBayes.csv"
