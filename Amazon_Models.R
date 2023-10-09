library(tidymodels)
library(vroom) 
library(ggmosaic) # For plotting

# Reading in the Data
AEAC_Train <- vroom("Amazon_AEAC_Kaggle/train.csv")
AEAC_Test <- vroom("Amazon_AEAC_Kaggle/test.csv")

AEAC_Train$ACTION = as.factor(AEAC_Train$ACTION)

AEAC_recipe <- recipe(ACTION ~., data=AEAC_Train) %>% 
  step_mutate_at(all_numeric_predictors(), fn= factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_dummy(all_nominal_predictors())# %>% 
#  step_lencode_mixed(all_nominal_predictors(), outcome= vars(ACTION))
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
                              type= "prob") # "class" or "prob" (see doc)
### FOR CLASS
log_submission <- amazon_log_predictions %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_class)

### FOR PROB
log_submission <- amazon_log_predictions %>% 
  mutate(id = row_number()) %>% 
  rename(Action = .pred_1) %>% 
  select(3,2)

vroom_write(x=log_submission, file="Amazon_AEAC_Kaggle/LogRegression.csv", delim=",")
