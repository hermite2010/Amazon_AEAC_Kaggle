library(tidymodels)
library(embed) # For target encoding
library(vroom) 
library(parallel)
library(ggmosaic) # For plotting

# Reading in the Data
AEAC_Train <- vroom("train.csv") #"Amazon_AEAC_Kaggle/train.csv" for local
AEAC_Test <- vroom("test.csv") #"Amazon_AEAC_Kaggle/test.csv" for local

AEAC_Train$ACTION = as.factor(AEAC_Train$ACTION)

AEAC_recipe <- recipe(ACTION ~., data=AEAC_Train) %>% 
  step_mutate_at(all_numeric_predictors(), fn= factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
#  step_dummy(all_nominal_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome= vars(ACTION))
# Try step_lencode_bayes() in the future

prep <- prep(AEAC_recipe)
baked_data <- bake(prep, new_data=AEAC_Train)


# Logistic Regression -----------------------------------------------------

log_reg_mod <- logistic_reg() %>% #Type of model
  set_engine("glm")

amazon_log_workflow <- workflow() %>%
add_recipe(AEAC_recipe) %>%
add_model(log_reg_mod) %>%
fit(data = AEAC_Train) # Fit the workflow

amazon_log_predictions <- predict(amazon_log_workflow,
                              new_data=AEAC_Test,
                              type= "prob") #%>% # "class" which uses cutoff as .5 or "prob" (see doc)
  #mutate(ACTION = ifelse(.pred_1> C,1,0)) #REPLACE C WITH WHATEVER CUTOFF I WANT

### FOR CLASS
log_submission <- amazon_log_predictions %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_class)

### FOR PROB
log_submission <- amazon_log_predictions %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_1) %>% 
  select(3,2)

vroom_write(x=log_submission, file="LogRegression.csv", delim=",") #"Amazon_AEAC_Kaggle/LogRegression.csv"


# Penalized Logistic Regression -------------------------------------------


pen_log_model <- logistic_reg(mixture=tune(),
                       penalty=tune()) %>% #Type of model
  set_engine("glmnet")

pen_log_workflow <- workflow() %>%
  add_recipe(AEAC_recipe) %>%
  add_model(pen_log_model)

## Grid of values to tune over
pen_log_tuning_grid <- grid_regular(penalty(),
                            mixture(), # Always bewteen 0 and 1
                            levels = 3) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(AEAC_Train, v = 3, repeats=1)

## Run the CV
CV_results <- pen_log_workflow %>%
  tune_grid(resamples=folds,
            grid=pen_log_tuning_grid,
            metrics=metric_set(roc_auc)) 

## Find Best Tuning Parameters
pen_log_bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- pen_log_workflow %>%
  finalize_workflow(pen_log_bestTune) %>%
  fit(data=AEAC_Train)

## Predict
pen_log_preds <- final_wf %>%
  predict(new_data = AEAC_Test, type="prob") %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_1) %>% 
  select(3,2)

vroom_write(x=pen_log_preds, file="PenLogRegression.csv", delim=",") #"Amazon_AEAC_Kaggle/PenLogRegression.csv"



