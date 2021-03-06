# This script provides functions to implement the k-NN algorithm
# for any distance measure provided by the package philentropy
# contains also few functions to make and asses predictions with naive Bayes
# and random Forest algorithm


  require(philentropy)
  require(tidyverse)
  require(tibble)
  require(e1071)
  require(coxed)
  require(randomForest)
  
# helper functions -----
  
  calculate_pred_stats <- function(prediction_outcome_tbl) {
    
    ## classifies the prediction results as a tp (true positive), tn (true negative)
    ## fp (false positive), fn (false negative), out of bag (oob) and correct.
    ## calculates prediction stats: sensitivity, specificity
    ## error and correct rate - for a binary response
    ## For a factor response, only the correct/oob classification is performed
    
    if(all(levels(factor(prediction_outcome_tbl$pred_outcome)) == c(0, 1))) {
      
      class_tbl <- prediction_outcome_tbl %>% 
        mutate(pred_outcome = as.numeric(as.character(pred_outcome)), 
               true_outcome = as.numeric(as.character(true_outcome))) %>% 
        mutate(tp = (pred_outcome == 1 & true_outcome == 1), 
               tn = (pred_outcome == 0 & true_outcome == 0), 
               fp = (pred_outcome == 1 & true_outcome == 0), 
               fn = (pred_outcome == 0 & true_outcome == 1), 
               oob = fp|fn, 
               correct = tp|tn)
      
      stat_tbl <- tibble(Se = sum(class_tbl$tp)/sum(class_tbl$true_outcome), 
                         Sp = sum(class_tbl$tn)/(nrow(class_tbl) - sum(class_tbl$true_outcome)), 
                         error_rate = sum(class_tbl$oob)/nrow(class_tbl), 
                         correct_rate = sum(class_tbl$correct)/nrow(class_tbl)) %>% 
        mutate(youden_j = Se + Sp - 1)
      
    } else {
      
      class_tbl <- prediction_outcome_tbl %>% 
        mutate(correct = (pred_outcome == true_outcome), 
               oob = (pred_outcome != true_outcome))
      
      stat_tbl <- tibble(error_rate = sum(class_tbl$oob)/nrow(class_tbl), 
                         correct_rate = sum(class_tbl$correct)/nrow(class_tbl))
      
    }
    
    return(list(result_class = class_tbl, 
                stats = stat_tbl))
      
  }
  
# naive Bayes and random forest prediction functions ------
  
  naive_bayes <- function(train, test_vec, outcome, type = 'class', ...) {
    
    ## makes prediction using naiveBayes() function provided by e1071 package
    ## handles both single- and mult-row data frames. ... are argmunents passed on
    ## to the mother naiveBayes function
    
    test_formula <- paste(outcome, '~.', sep = '') %>% 
      as.formula
    
    ## training the Bayes model
    
    trained_model <- naiveBayes(formula = test_formula, 
                                data = train, ...)
    
    ## prediction
    
    prediction <- predict(trained_model, 
                          newdata = test_vec, 
                          type = type)
    
    pred_tbl <- tibble(prediction = prediction, 
                       test_set_id = rownames(test_vec)) %>% 
      set_names(c(outcome, 
                  'test_set_id'))
    
    return(pred_tbl)
    
  }
  
  random_forest <- function(train, test_vec, outcome, return_model = F, ...) {
    
    ## makes prediction using randomForest() function provided by randomForest package
    ## handles both single- and mult-row data frames. ... are argmunents passed on
    ## to the mother randomForest function
    ## can return the model in addtion to the predictions
    
    test_formula <- paste(outcome, '~.', sep = '') %>% 
      as.formula
    
    ## training the Bayes model
    
    trained_model <- randomForest(formula = test_formula, 
                                  data = train, ...)
    
    ## prediction
    
    prediction <- predict(trained_model, 
                          newdata = test_vec)
    
    pred_tbl <- tibble(prediction = prediction, 
                       test_set_id = rownames(test_vec)) %>% 
      set_names(c(outcome, 
                  'test_set_id'))
    
    if(return_model) {
      
      return(list(model = trained_model, 
                  prediction = pred_tbl))
      
    } else {
      
      return(pred_tbl)
      
    }
    
  }

# knn prediction -----  

  knn <- function(train, test_vec, outcome, k = 5,  method = 'jaccard', predict = T, 
                  kernel_fun = function(x) 1, ...){
    
    ## chooses the nearest neighbors or 
    ## predicts the feature (outcome) value for the given vector of features based on the k-nearest neighbors
    ## the solution modified from: 
    ## https://stackoverflow.com/questions/50428127/implementing-knn-with-different-distance-metrics-using-r
    ## works for any factorized variables. Kernel function specified by the user allows for weighted voting
    
    
    
    ## handles multiple-row data frame as a test vector by recursion
    
    
    if(any(class(test_vec) == 'data.frame') & nrow(test_vec) > 1) {
      
      pred_results <- 1:nrow(test_vec) %>% 
        map_dfr(function(x) knn(train = train, 
                                test_vec = test_vec[x, ], 
                                k = k, 
                                outcome = outcome, 
                                method = method,
                                kernel_fun = kernel_fun, 
                                predict = T, 
                                ...))
      
      pred_results <- pred_results %>% 
        mutate(test_set_id = rownames(test_vec))
      
      return(pred_results)
      
    }
    
    ## distance calculation
    
    test_wo_outcome <- test_vec[, names(test_vec) != outcome]
    
    train_wo_outcome <- train[, names(train) != outcome]
    
    
    if(ncol(test_vec) != ncol(train)) {
      
      stop('the test and treining data sets must contain the same number of variables')
      
    }
    
    dist_storage <- 1:nrow(train) %>% 
      map_dfr(function(x) tibble(id_train = x, 
                                 distance = distance(rbind(test_wo_outcome, 
                                                           train_wo_outcome[x, ]), 
                                                     method = method, 
                                                     mute.message = T, ...)))
    
    dist_storage <- dist_storage %>% 
      mutate(outcome = train[[outcome]]) %>% 
      set_names(c('ID_train', 
                  'distance', 
                  outcome)) %>% 
      mutate(weighted_vote = kernel_fun(distance))
    
    ## identificaiton of the NNs
    
    neigh <- dist_storage %>% 
      top_n(n = k, 
            desc(distance))
    
    if(!predict) {
      
      return(neigh)
      
    } else {
      
      ## voting 
      
      voting_sum_tbl <- neigh %>% 
        group_by(.data[[outcome]]) %>% 
        dplyr::summarise(vote_sum = sum(weighted_vote)) %>%
        filter(vote_sum == max(vote_sum))
      
      if(nrow(voting_sum_tbl) == 1) {
        
        return(voting_sum_tbl)
        
      } else {
        
        tie_break <- sample(1:nrow(voting_sum_tbl), 
                            size = 1)
        
        return(voting_sum_tbl[1, ])
        
        
      }
      
    }
    
  }
  
# accuracy testing ------
  
  check_accuracy <- function(train, test_vec, outcome, pred_fun = 'knn', regression = F, detailed = F, ...) {
    
    ## compares the predictions by the given predicting function with the real results
    ## returns either a summary table or detailed results. ... specifies additional arguments passed
    ## to the predicting function.
    ## For regression predictions: a simple table with the predicted and actual value is returned
    
    if(is.null(rownames(train)) | is.null(rownames(train))) {
      
      stop('The function requires a data frame with specified rownames')
      
    }
    
    if(nrow(test_vec) == 1) {
      
      stop('To calculate prediction statistics, a test data set with more observations is required (nrow > 1)')
      
    }
    
    ## making predictions

    if(pred_fun == 'knn') {
      
      prediction <- knn(train = train, 
                        test_vec = test_vec, 
                        outcome = outcome, ...) %>% 
        set_names(c('pred_outcome', 
                    'voting_sum',  
                    'test_set_id'))
      
    } else if(pred_fun == 'naive_bayes') {
      
      prediction <- naive_bayes(train = train, 
                                test_vec = test_vec, 
                                outcome = outcome, ...) %>% 
        set_names(c('pred_outcome', 
                    'test_set_id'))
      
    } else if(pred_fun == 'random_forest') {
      
      prediction <- random_forest(train = train, 
                                  test_vec = test_vec, 
                                  outcome = outcome, 
                                  return_model = F, ...) %>% 
        set_names(c('pred_outcome', 
                    'test_set_id'))
      
    } else {
      
      stop('Please specify the prediction function. Currently: knn, naive_bayes or random_forest')
      
    }
    
   ## comparing them with the true outcome
    
    real_outcome <- test_vec %>% 
      rownames_to_column('test_set_id') %>% 
      select(all_of(c('test_set_id', 
                      outcome))) %>% 
      set_names(c('test_set_id', 
                  'true_outcome'))
    
    pred_stats <- left_join(prediction, 
                            real_outcome, 
                            by = 'test_set_id')
    
    if(regression) {
      
      pred_stats <- pred_stats %>% 
        mutate(y = true_outcome, 
               .fitted = pred_outcome, 
               .resid = pred_outcome - true_outcome, 
               .sq.resid = .resid ^ 2, 
               .std.resid = scale(.resid)[, 1], 
               .sq.std.resid = .std.resid ^ 2, 
               .candidate_missfit = ifelse(abs(.std.resid) > qnorm(0.975), 
                                           'yes', 
                                           'no'))
      
      measures <- list(mse = mean(pred_stats$.sq.resid), 
                       mae = mean(abs(pred_stats$.resid)), 
                       corr_pearson = cor(pred_stats$y, 
                                          pred_stats$.fitted, 
                                          method = 'pearson'), 
                       corr_spearman = cor(pred_stats$y, 
                                          pred_stats$.fitted, 
                                          method = 'spearman'), 
                       rsq = 1 - mean(pred_stats$.sq.resid)/var(pred_stats$y), 
                       n_complete = nrow(pred_stats)) %>% 
        as_tibble
    
      return(list(fit = pred_stats, 
                  measures = measures))
      
    }
    
    pred_stats <- pred_stats %>% 
      calculate_pred_stats

    if(detailed) {
      
      return(list(pred_results = pred_stats$result_class, 
                  stats = pred_stats$stats))
      
    } else {
      
      return(pred_stats$stats)
      
    }
    
  }
  
# random generation of the training/test splits of the given data set ----
  
  make_splits <- function(inp_tbl, nrow_train, n_splits = 100) {
    
    ## creates a series of train/test data sets out of the given table
    ## the size of the training data set can be specified by the user
    ## for optimal performance and compatibility with the knn prediction
    ## and accuracy testing functions, a data frame with specified row names
    ## is required
    
    if(is.null(rownames(inp_tbl))) {
      
      stop('The function requires a data frame with specified rownames')
      
    }
    
    tbl_len <- nrow(inp_tbl)
    
    split_ids <- paste('split', 1:n_splits, sep = '_') %>% 
      map(function(x) sample(1:tbl_len, nrow_train)) %>% 
      set_names(paste('split', 1:n_splits, sep = '_'))
    
    split_lst <- split_ids %>% 
      map(function(x) list(train = inp_tbl[x, ], 
                           test = inp_tbl[!(1:tbl_len) %in% x, ]))
    
    return(split_lst)
    
  }
  
# Serial testing of the prediction quality with multiple random training/test splits -----

  test_accuracy <- function(boot_split_lst, outcome, pred_fun = 'knn', 
                            generate_random = F, exp_estimate = mean, ci_method = 'percentile', 
                            .parallel = T, ...) {
    
    ## calculates sensitivity and specificity of the k-NN pr naiveBayes approach
    ## given the list of training/test. Generate_random: a set of coin-toss predictions
    ## in the for the test sets is generated and sensitivity, specificity, correctness and error
    ## rate calculated and the significance of the k-NN prediction stats (better than random)
    ## calculated with Wilcoxon test ... are arguments passed to the predicting function.
    ## The ci_methods defining the way, CI's are calculated encompass the standard percentile method
    ## and bca provided by coxed package. exp_estimate defines how the expected value should
    ## be computed: e.g. as median or mean of the bootstraped results.
    
    if(.parallel) {
      
      require(furrr)
      
      plan('multisession')
      
      qc_results <- boot_split_lst %>% 
        future_map_dfr(function(x) check_accuracy(train = x$train,
                                                  test_vec = x$test,  
                                                  outcome = outcome, 
                                                  pred_fun = pred_fun, 
                                                  detailed = F, ...), 
                       .options = furrr_options(seed = T))
      
      plan('sequential')
      
    } else {
      
      qc_results <- boot_split_lst %>% 
        map_dfr(function(x) check_accuracy(train = x$train,
                                           test_vec = x$test,  
                                           outcome = outcome, 
                                           pred_fun = pred_fun, 
                                           detailed = F, ...))
      
    }
    
    if(ci_method == 'percentile') {
      
      ci_fun <- function(x) quantile(x, c(0.025, 0.975))
      
    } else if(ci_method == 'bca') {
      
      ci_fun <- bca
      
    } else {
      
      stop('Undefined CI calculation method. Currently supported: percentile and bca')
      
    }
    
    summary_qc <- qc_results %>% 
      map_dfr(function(x) c(exp_estimate(x), 
                            ci_fun(x)) %>% 
                set_names(c('expected', 'lower_ci', 'upper_ci'))) %>% 
      mutate(stat = names(qc_results))
    
    if(!generate_random) {
      
      return(list(boot_results = qc_results, 
                  summary = summary_qc))
      
    } else {
      
      rand_results <- boot_split_lst %>% 
        map(function(x) tibble(true_outcome = x$test[[outcome]], 
                               pred_outcome = sample(levels(factor(x$test[[outcome]])), 
                                                     size = nrow(x$test), 
                                                     replace = T))) %>% 
        map_dfr(function(x) calculate_pred_stats(x)$stats)
      
      rand_summary_qc <- rand_results %>% 
        map_dfr(function(x) c(exp_estimate(x), 
                              ci_fun(x)) %>% 
                  set_names(c('expected', 'lower_ci', 'upper_ci'))) %>% 
        mutate(stat = names(rand_results))
      
      ## testing whether the Se, Sp and rate values are better than random: Wilcoxon test
      
      wilcox_test <- map2(qc_results, 
                          rand_results, 
                          function(x, y) suppressWarnings(wilcox.test(x, y))) %>% 
        map_dfr(function(x) tibble(W = x$statistic, 
                                   p_value = x$p.value)) %>% 
        mutate(stat = names(qc_results))
      
      return(list(boot_results = qc_results, 
                  rand_results = rand_results, 
                  summary = summary_qc, 
                  rand_summary = rand_summary_qc, 
                  significance = wilcox_test) %>% 
               map(mutate, 
                   pred_method = pred_fun))
      
    }
    
  }
  
# END -----