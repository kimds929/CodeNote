# 1. Supervised learning
    # 1.1. Linear Models
    # 1.2. Linear and Quadratic Discriminant Analysis
    # 1.3. Kernel ridge regression
    # 1.4. Support Vector Machines
    # 1.5. Stochastic Gradient Descent
    # 1.6. Nearest Neighbors
    # 1.7. Gaussian Processes
    # 1.8. Cross decomposition
    # 1.9. Naive Bayes
    # 1.10. Decision Trees
    # 1.11. Ensemble methods
    # 1.12. Multiclass and multilabel algorithms
    # 1.13. Feature selection
    # 1.14. Semi-Supervised
    # 1.15. Isotonic regression
    # 1.16. Probability calibration
    # 1.17. Neural network models (supervised)


# 2. Unsupervised learning
    # 2.1. Gaussian mixture models
    # 2.2. Manifold learning
    # 2.3. Clustering
    # 2.4. Biclustering
    # 2.5. Decomposing signals in components (matrix factorization problems)
    # 2.6. Covariance estimation
    # 2.7. Novelty and Outlier Detection
    # 2.8. Density Estimation
    # 2.9. Neural network models (unsupervised)


# 3. Model selection and evaluation
    # 3.1. Cross-validation: evaluating estimator performance
    # 3.2. Tuning the hyper-parameters of an estimator
    # 3.3. Metrics and scoring: quantifying the quality of predictions
    # 3.4. Model persistence
    # 3.5. Validation curves: plotting scores to evaluate models


# 4. Inspection
    # 4.1. Partial dependence plots
    # 4.2. Permutation feature importance


# 5. Visualizations
    # 5.1. Available Plotting Utilities


# 6. Dataset transformations
    # 6.1. Pipelines and composite estimators
    # 6.2. Feature extraction
    # 6.3. Preprocessing data
    # 6.4. Imputation of missing values
    # 6.5. Unsupervised dimensionality reduction
    # 6.6. Random Projection
    # 6.7. Kernel Approximation
    # 6.8. Pairwise metrics, Affinities and Kernels
    # 6.9. Transforming the prediction target (y)


# 7. Dataset loading utilities
    # 7.1. General dataset API
    # 7.2. Toy datasets
    # 7.3. Real world datasets
    # 7.4. Generated datasets
    # 7.5. Loading other datasets


# 8. Computing with scikit-learn
    # 8.1. Strategies to scale computationally: bigger data
    # 8.2. Computational Performance
    # 8.3. Parallelism, resource management, and configuration







# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------



# [선형 모델] sklearn.linear_model --------------------------------------------------------------------------
    # sklearn.linear_model.ARDRegression
    # sklearn.linear_model.BayesianRidge
    # sklearn.linear_model.ElasticNet
    # sklearn.linear_model.ElasticNetCV
    # sklearn.linear_model.enet_path
    # sklearn.linear_model.GammaRegressor
    # sklearn.linear_model.HuberRegressor
    # sklearn.linear_model.Lars
    # sklearn.linear_model.lars_path
    # sklearn.linear_model.lars_path_gram
    # sklearn.linear_model.LarsCV
    # sklearn.linear_model.Lasso
    # sklearn.linear_model.lasso_path
    # sklearn.linear_model.LassoCV
    # sklearn.linear_model.LassoLars
    # sklearn.linear_model.LassoLarsCV
    # sklearn.linear_model.LassoLarsIC
    # sklearn.linear_model.LinearRegression
    # sklearn.linear_model.LogisticRegression
    # sklearn.linear_model.LogisticRegressionCV
    # sklearn.linear_model.MultiTaskElasticNet
    # sklearn.linear_model.MultiTaskElasticNetCV
    # sklearn.linear_model.MultiTaskLasso
    # sklearn.linear_model.MultiTaskLassoCV
    # sklearn.linear_model.orthogonal_mp
    # sklearn.linear_model.orthogonal_mp_gram
    # sklearn.linear_model.OrthogonalMatchingPursuit
    # sklearn.linear_model.OrthogonalMatchingPursuitCV
    # sklearn.linear_model.PassiveAggressiveClassifier
    # sklearn.linear_model.PassiveAggressiveRegressor
    # sklearn.linear_model.Perceptron
    # sklearn.linear_model.PoissonRegressor
    # sklearn.linear_model.RANSACRegressor
    # sklearn.linear_model.Ridge
    # sklearn.linear_model.ridge_regression
    # sklearn.linear_model.RidgeClassifier
    # sklearn.linear_model.RidgeClassifierCV
    # sklearn.linear_model.RidgeCV
    # sklearn.linear_model.SGDClassifier
    # sklearn.linear_model.SGDRegressor
    # sklearn.linear_model.TheilSenRegressor
    # sklearn.linear_model.TweedieRegressor




# [교차 분해] sklearn.cross_decomposition --------------------------------------------------------------------------
    # sklearn.cross_decomposition.CCA
    # sklearn.cross_decomposition.PLSCanonical
    # sklearn.cross_decomposition.PLSRegression
    # sklearn.cross_decomposition.PLSSVD



# [앙상블] sklearn.ensemble  --------------------------------------------------------------------------
    # sklearn.ensemble.AdaBoostClassifier
    # sklearn.ensemble.AdaBoostRegressor
    # sklearn.ensemble.BaggingClassifier
    # sklearn.ensemble.BaggingRegressor
    # sklearn.ensemble.ExtraTreesClassifier
    # sklearn.ensemble.ExtraTreesRegressor
    # sklearn.ensemble.GradientBoostingClassifier
    # sklearn.ensemble.GradientBoostingRegressor
    # sklearn.ensemble.HistGradientBoostingClassifier
    # sklearn.ensemble.HistGradientBoostingRegressor
    # sklearn.ensemble.IsolationForest
    # sklearn.ensemble.RandomForestClassifier
    # sklearn.ensemble.RandomForestRegressor
    # sklearn.ensemble.RandomTreesEmbedding
    # sklearn.ensemble.StackingClassifier
    # sklearn.ensemble.StackingRegressor
    # sklearn.ensemble.VotingClassifier
    # sklearn.ensemble.VotingRegressor



# [변수 선택] sklearn.feature_selection --------------------------------------------------------------------------
    # sklearn.feature_selection.chi2
    # sklearn.feature_selection.f_classif
    # sklearn.feature_selection.f_regression
    # sklearn.feature_selection.GenericUnivariateSelect
    # sklearn.feature_selection.mutual_info_classif
    # sklearn.feature_selection.mutual_info_regression
    # sklearn.feature_selection.RFE
    # sklearn.feature_selection.RFECV
    # sklearn.feature_selection.SelectFdr
    # sklearn.feature_selection.SelectFpr
    # sklearn.feature_selection.SelectFromModel
    # sklearn.feature_selection.SelectFwe
    # sklearn.feature_selection.SelectKBest
    # sklearn.feature_selection.SelectorMixin
    # sklearn.feature_selection.SelectPercentile
    # sklearn.feature_selection.VarianceThreshold



# [데이터셋 선택] model_selection --------------------------------------------------------------------------
    # sklearn.model_selection.check_cv
    # sklearn.model_selection.cross_val_predict
    # sklearn.model_selection.cross_val_score
    # sklearn.model_selection.cross_validate
    # sklearn.model_selection.fit_grid_point
    # sklearn.model_selection.GridSearchCV
    # sklearn.model_selection.GroupKFold
    # sklearn.model_selection.GroupShuffleSplit
    # sklearn.model_selection.KFold
    # sklearn.model_selection.learning_curve
    # sklearn.model_selection.LeaveOneGroupOut
    # sklearn.model_selection.LeaveOneOut
    # sklearn.model_selection.LeavePGroupsOut
    # sklearn.model_selection.LeavePOut
    # sklearn.model_selection.ParameterGrid
    # sklearn.model_selection.ParameterSampler
    # sklearn.model_selection.permutation_test_score
    # sklearn.model_selection.PredefinedSplit
    # sklearn.model_selection.RandomizedSearchCV
    # sklearn.model_selection.RepeatedKFold
    # sklearn.model_selection.RepeatedStratifiedKFold
    # sklearn.model_selection.ShuffleSplit
    # sklearn.model_selection.StratifiedKFold
    # sklearn.model_selection.StratifiedShuffleSplit
    # sklearn.model_selection.TimeSeriesSplit
    # sklearn.model_selection.train_test_split
    # sklearn.model_selection.validation_curve



# [모델 평가] sklearn.metrics : # sklearn.metrics.SCORERS.keys()  --------------------------------------------------------------------------

    # sklearn.metrics.pairwise.distance_metrics
    # sklearn.metrics.pairwise.kernel_metrics
    # sklearn.metrics.cluster
    # sklearn.metrics.pairwise

    # sklearn.metrics.accuracy_score
    # sklearn.metrics.adjusted_mutual_info_score
    # sklearn.metrics.adjusted_rand_score
    # sklearn.metrics.auc
    # sklearn.metrics.average_precision_score
    # sklearn.metrics.balanced_accuracy_score
    # sklearn.metrics.brier_score_loss
    # sklearn.metrics.calinski_harabasz_score
    # sklearn.metrics.check_scoring
    # sklearn.metrics.classification_report
    # sklearn.metrics.cluster.contingency_matrix
    # sklearn.metrics.cohen_kappa_score
    # sklearn.metrics.completeness_score
    # sklearn.metrics.confusion_matrix
    # sklearn.metrics.ConfusionMatrixDisplay
    # sklearn.metrics.consensus_score
    # sklearn.metrics.coverage_error
    # sklearn.metrics.davies_bouldin_score
    # sklearn.metrics.dcg_score
    # sklearn.metrics.explained_variance_score
    # sklearn.metrics.f1_score
    # sklearn.metrics.fbeta_score
    # sklearn.metrics.fowlkes_mallows_score
    # sklearn.metrics.get_scorer
    # sklearn.metrics.hamming_loss
    # sklearn.metrics.hinge_loss
    # sklearn.metrics.homogeneity_completeness_v_measure
    # sklearn.metrics.homogeneity_score
    # sklearn.metrics.jaccard_score
    # sklearn.metrics.label_ranking_average_precision_score
    # sklearn.metrics.label_ranking_loss
    # sklearn.metrics.log_loss          # cross_entropy
    # sklearn.metrics.make_scorer
    # sklearn.metrics.matthews_corrcoef
    # sklearn.metrics.max_error
    # sklearn.metrics.mean_absolute_error
    # sklearn.metrics.mean_gamma_deviance
    # sklearn.metrics.mean_poisson_deviance
    # sklearn.metrics.mean_squared_error
    # sklearn.metrics.mean_squared_log_error
    # sklearn.metrics.mean_tweedie_deviance
    # sklearn.metrics.median_absolute_error
    # sklearn.metrics.multilabel_confusion_matrix
    # sklearn.metrics.mutual_info_score
    # sklearn.metrics.ndcg_score
    # sklearn.metrics.normalized_mutual_info_score
    # sklearn.metrics.pairwise.additive_chi2_kernel
    # sklearn.metrics.pairwise.chi2_kernel
    # sklearn.metrics.pairwise.cosine_distances
    # sklearn.metrics.pairwise.cosine_similarity
    # sklearn.metrics.pairwise.euclidean_distances
    # sklearn.metrics.pairwise.haversine_distances
    # sklearn.metrics.pairwise.laplacian_kernel
    # sklearn.metrics.pairwise.linear_kernel
    # sklearn.metrics.pairwise.manhattan_distances
    # sklearn.metrics.pairwise.nan_euclidean_distances
    # sklearn.metrics.pairwise.paired_cosine_distances
    # sklearn.metrics.pairwise.paired_distances
    # sklearn.metrics.pairwise.paired_euclidean_distances
    # sklearn.metrics.pairwise.paired_manhattan_distances
    # sklearn.metrics.pairwise.pairwise_kernels
    # sklearn.metrics.pairwise.polynomial_kernel
    # sklearn.metrics.pairwise.rbf_kernel
    # sklearn.metrics.pairwise.sigmoid_kernel
    # sklearn.metrics.pairwise_distances
    # sklearn.metrics.pairwise_distances_argmin
    # sklearn.metrics.pairwise_distances_argmin_min
    # sklearn.metrics.pairwise_distances_chunked
    # sklearn.metrics.plot_confusion_matrix
    # sklearn.metrics.plot_precision_recall_curve
    # sklearn.metrics.plot_roc_curve
    # sklearn.metrics.precision_recall_curve
    # sklearn.metrics.precision_recall_fscore_support
    # sklearn.metrics.precision_score
    # sklearn.metrics.PrecisionRecallDisplay
    # sklearn.metrics.r2_score
    # sklearn.metrics.recall_score
    # sklearn.metrics.roc_auc_score
    # sklearn.metrics.roc_curve
    # sklearn.metrics.RocCurveDisplay
    # sklearn.metrics.silhouette_samples
    # sklearn.metrics.silhouette_score
    # sklearn.metrics.v_measure_score
    # sklearn.metrics.zero_one_loss



# [데이터 전처리] sklearn.preprocessing --------------------------------------------------------------------------
    # sklearn.preprocessing.add_dummy_feature
    # sklearn.preprocessing.binarize
    # sklearn.preprocessing.Binarizer
    # sklearn.preprocessing.FunctionTransformer
    # sklearn.preprocessing.KBinsDiscretizer
    # sklearn.preprocessing.KernelCenterer
    # sklearn.preprocessing.label_binarize
    # sklearn.preprocessing.LabelBinarizer
    # sklearn.preprocessing.LabelEncoder
    # sklearn.preprocessing.maxabs_scale
    # sklearn.preprocessing.MaxAbsScaler
    # sklearn.preprocessing.minmax_scale
    # sklearn.preprocessing.MinMaxScaler
    # sklearn.preprocessing.MultiLabelBinarizer
    # sklearn.preprocessing.normalize
    # sklearn.preprocessing.Normalizer
    # sklearn.preprocessing.OneHotEncoder
    # sklearn.preprocessing.OrdinalEncoder
    # sklearn.preprocessing.PolynomialFeatures
    # sklearn.preprocessing.power_transform
    # sklearn.preprocessing.PowerTransformer
    # sklearn.preprocessing.quantile_transform
    # sklearn.preprocessing.QuantileTransformer
    # sklearn.preprocessing.robust_scale
    # sklearn.preprocessing.RobustScaler
    # sklearn.preprocessing.scale
    # sklearn.preprocessing.StandardScaler








# [유틸리티] sklearn.utils -------------------------------------------------------------------------
    # sklearn.utils._safe_indexing
    # sklearn.utils.all_estimators
    # sklearn.utils.arrayfuncs.min_pos
    # sklearn.utils.as_float_array
    # sklearn.utils.assert_all_finite
    # sklearn.utils.Bunch
    # sklearn.utils.check_array
    # sklearn.utils.check_consistent_length
    # sklearn.utils.check_random_state
    # sklearn.utils.check_scalar
    # sklearn.utils.check_X_y
    # sklearn.utils.class_weight.compute_class_weight
    # sklearn.utils.class_weight.compute_sample_weight
    # sklearn.utils.deprecated
    # sklearn.utils.estimator_checks.check_estimator
    # sklearn.utils.estimator_checks.parametrize_with_checks
    # sklearn.utils.estimator_html_repr
    # sklearn.utils.extmath.density
    # sklearn.utils.extmath.fast_logdet
    # sklearn.utils.extmath.randomized_range_finder
    # sklearn.utils.extmath.randomized_svd
    # sklearn.utils.extmath.safe_sparse_dot
    # sklearn.utils.extmath.weighted_mode
    # sklearn.utils.gen_even_slices
    # sklearn.utils.graph.single_source_shortest_path_length
    # sklearn.utils.graph_shortest_path.graph_shortest_path
    # sklearn.utils.indexable
    # sklearn.utils.metaestimators.if_delegate_has_method
    # sklearn.utils.multiclass.is_multilabel
    # sklearn.utils.multiclass.type_of_target
    # sklearn.utils.multiclass.unique_labels
    # sklearn.utils.murmurhash3_32
    # sklearn.utils.parallel_backend
    # sklearn.utils.random.sample_without_replacement
    # sklearn.utils.register_parallel_backend
    # sklearn.utils.resample
    # sklearn.utils.safe_indexing
    # sklearn.utils.safe_mask
    # sklearn.utils.safe_sqr
    # sklearn.utils.shuffle
    # sklearn.utils.sparsefuncs.incr_mean_variance_axis
    # sklearn.utils.sparsefuncs.inplace_column_scale
    # sklearn.utils.sparsefuncs.inplace_csr_column_scale
    # sklearn.utils.sparsefuncs.inplace_row_scale
    # sklearn.utils.sparsefuncs.inplace_swap_column
    # sklearn.utils.sparsefuncs.inplace_swap_row
    # sklearn.utils.sparsefuncs.mean_variance_axis
    # sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1
    # sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2
    # sklearn.utils.validation.check_is_fitted
    # sklearn.utils.validation.check_memory
    # sklearn.utils.validation.check_symmetric
    # sklearn.utils.validation.column_or_1d
    # sklearn.utils.validation.has_fit_parameter