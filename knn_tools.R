# This script provides functions to implement the k-NN algorithm
# for any distance measure provided by the package philentropy

  require(philentropy)
  require(tidyverse)
  require(tibble)
  
# helper functions -----
  
  calculate_pred_stats <- function(prediction_outcome_tbl) {
    
    ## classifies the prediction results as a tp (true positive), tn (true negative)
    ## fp (false positive), fn (false negative), out of bag (oob) and correct.
    ## calculates prediction stats: sensitivity, specificity
    ## error and correct rate
    
    class_tbl <- prediction_outcome_tbl %>% 
      mutate(tp = (pred_outcome == 1 & true_outcome == 1), 
             tn = (pred_outcome == 0 & true_outcome == 0), 
             fp = (pred_outcome == 1 & true_outcome == 0), 
             fn = (pred_outcome == 0 & true_outcome == 1), 
             oob = fp|fn, 
             correct = tp|tn)
    
    stat_tbl <- tibble(Se = sum(class_tbl$tp)/sum(class_tbl$true_outcome), 
                       Sp = sum(class_tbl$tn)/(nrow(class_tbl) - sum(class_tbl$true_outcome)), 
                       error_rate = sum(class_tbl$oob)/nrow(class_tbl), 
                       correct_rate = sum(class_tbl$correct)/nrow(class_tbl))
    
    return(list(result_class = class_tbl, 
                stats = stat_tbl))
      
    
  }

# knn porediction and testing functions -----  

  knn <- function(train, test_vec, k, outcome,  method = 'jaccard', predict = T, include_poisson = F, ...){
    
    ## chooses the nearest neighbors or 
    ## predicts the feature (outcome) value for the given vector of features based on the k-nearest neighbors
    ## the solution modified from: 
    # https://stackoverflow.com/questions/50428127/implementing-knn-with-different-distance-metrics-using-r
    ## works for binary variables. If needed, Poisson test may be made on the voting results
    ## handles multiple data frame as a test vector by recursion
    
    
    if(any(class(test_vec) == 'data.frame') & nrow(test_vec) > 1) {
      
      pred_results <- 1:nrow(test_vec) %>% 
        map_dfr(function(x) knn(train = train, 
                                test_vec = test_vec[x, ], 
                                k = k, 
                                outcome = outcome, 
                                method = method, 
                                predict = predict, 
                                include_poisson = include_poisson, 
                                ...))
      
      pred_results <- pred_results %>% 
        mutate(test_set_id = rownames(test_vec))
      
      return(pred_results)
      
    }
    
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
                  outcome))
    
    neigh <- dist_storage %>% 
      top_n(n = k, 
            desc(distance))
    
    if(!predict) {
      
      return(neigh)
      
    } else {
      
      ## voting
      
      decision_stat <- mean(neigh[[outcome]])
      
      prediction <- ifelse(decision_stat == 0.5, 
                           sample(c(1, 0), 1), 
                           ifelse(decision_stat > 0.5, 1, 0))
      
      if(include_poisson) {
        
        ## voting certainty obtained by Poisson test
        
        pred_certain <- poisson.test(c(sum(neigh[[outcome]]), 
                                       length(neigh[[outcome]]) - sum(neigh[[outcome]])))
        
        pred_tbl <- tibble(prediction = prediction, 
                           decision_stat = decision_stat, 
                           p_val = pred_certain$p.value) %>% 
          set_names(c(outcome, 
                      'voting_result', 
                      'p_value'))
        
      } else {
        
        pred_tbl <- tibble(prediction = prediction, 
                           decision_stat = decision_stat) %>% 
          set_names(c(outcome, 
                      'voting_result'))
        
      }
      
      return(pred_tbl)
      
    }
    
  }
  
  check_accuracy <- function(train, test_vec, k, outcome, method = 'jaccard', detailed = F, ...) {
    
    ## compares the predictions with the real results
    ## returns either a summary table or detailed results
    
    if(is.null(rownames(train)) | is.null(rownames(train))) {
      
      stop('The function requires a data frame with specified rownames')
      
    }
    
    if(nrow(test_vec) == 1) {
      
      stop('To calculate prediction statistics, a test data set with more observation is required (nrow > 1)')
      
    }
    
    prediction <- knn(train = train, 
                      test_vec = test_vec, 
                      k = k, 
                      outcome = outcome, 
                      method = method, 
                      include_poisson = F, ...) %>% 
      set_names(c('pred_outcome', 
                  'voting_result',  
                  'test_set_id'))
    
    real_outcome <- test_vec %>% 
      rownames_to_column('test_set_id') %>% 
      select(all_of(c('test_set_id', 
                      outcome))) %>% 
      set_names(c('test_set_id', 
                  'true_outcome'))
    
    pred_stats <- left_join(prediction, 
                            real_outcome, 
                            by = 'test_set_id') %>% 
      calculate_pred_stats

    if(detailed) {
      
      return(pred_results = pred_stats$result_class, 
             stats = pred_stats$stats)
      
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
  
  test_knn <- function(boot_split_lst, k, outcome, method = 'jaccard', generate_random = F, .parallel = T, ...) {
    
    ## calculates sensitivity and specificity of the k-NN approach
    ## given the list of training/test. Generate_random: a set of coin-toss predictions
    ## in the for the test sets is generated and sensitivity, specificity, correctness and error
    ## rate calculated and the significance of the k-NN prediction stats (better than random)
    ## calculated with Wilcoxon test
    
    if(.parallel) {
      
      require(furrr)
      
      plan('multisession')
      
      qc_results <- boot_split_lst %>% 
        future_map_dfr(function(x) check_accuracy(train = x$train,
                                                  test_vec = x$test,  
                                                  k = k, 
                                                  outcome = outcome, 
                                                  method = method, ...), 
                       .options = furrr_options(seed = T))
      
      plan('sequential')
      
    } else {

      qc_results <- boot_split_lst %>% 
        map_dfr(function(x) check_accuracy(train = x$train,
                                           test_vec = x$test,  
                                           k = k, 
                                           outcome = outcome, ...))
      
    }
    
    summary_qc <- qc_results %>% 
      map_dfr(quantile, 
              c(0.5, 0.025, 0.975)) %>% 
      set_names(c('median', 'lower_ci', 'upper_ci')) %>% 
      mutate(stat = names(qc_results))
    
    if(!generate_random) {
      
      return(list(boot_results = qc_results, 
                  summary = summary_qc))
      
    } else {
      
      rand_results <- boot_split_lst %>% 
        map(function(x) tibble(true_outcome = x$test[[outcome]], 
                               pred_outcome = sample(c(0, 1), 
                                                     size = nrow(x$test), 
                                                     replace = T))) %>% 
        map_dfr(function(x) calculate_pred_stats(x)$stats)
      
      rand_summary_qc <- rand_results %>% 
        map_dfr(quantile, 
                c(0.5, 0.025, 0.975)) %>% 
        set_names(c('median', 'lower_ci', 'upper_ci')) %>% 
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
                  significance = wilcox_test))

    }
    
  }



  