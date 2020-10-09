columns_for_training = ["log_budget", "log_popularity", "log_runtime", "day_of_week", "year", "month", "week_of_year", "season",
                        "num_genres", "num_of_production_countries", "log_num_of_cast", "log_num_of_male_cast", "log_num_of_female_cast", "has_collection", 
                        "has_homepage", "has_tag", "is_english_language",
                       "log_num_of_crew", "log_num_of_male_crew", "log_num_of_female_crew",
                       "log_title_len", "log_overview_len", "log_tagline_len",
                       "log_num_of_directors", "log_num_of_producers", "log_num_of_editors", "log_num_of_art_crew", "log_num_of_sound_crew",
                       "log_num_of_costume_crew", "log_num_of_camera_crew", "log_num_of_visual_effects_crew", "log_num_of_lighting_crew",
                        "log_num_of_other_crew"]


 # adding isTopGenre_ columns for features before ML modeling
 columns_for_training.extend(dataset1.select(lambda col: col.startswith('isTopGenre_'), axis=1).columns.values)
 
 # adding isTopProductionCompany_ columns for features before ML modeling
 columns_for_training.extend(dataset1.select(lambda col: col.startswith('isTopProductionCompany_'), axis=1).columns.values)
 
 # adding isTopProductionCountry_ columns for features before ML modeling
 columns_for_training.extend(dataset1.select(lambda col: col.startswith('isTopProductionCountry_'), axis=1).columns.values)
 
 # adding has_top_actor_ columns for features before ML modeling
 columns_for_training.extend(dataset1.select(lambda col: col.startswith('has_top_actor_'), axis=1).columns.values)
 
 # adding has_top_keyword_ columns for features before ML modeling
 columns_for_training.extend(dataset1.select(lambda col: col.startswith('has_top_keyword_'), axis=1).columns.values)
 
 # adding has_top_director_ columns for features before ML modeling
 columns_for_training.extend(dataset1.select(lambda col: col.startswith('has_top_director_'), axis=1).columns.values)
 
 # adding has_top_producer_ columns for features before ML modeling
 columns_for_training.extend(dataset1.select(lambda col: col.startswith('has_top_producer_'), axis=1).columns.values) 


dataset1[columns_for_training].columns



X = dataset1[columns_for_training]
y = dataset1['log_revenue']


#                      Linear Regression



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)   

kfold_splits = 5

predictions_linear_regression_test = np.zeros(len(dataset))
num_fold = 0
oof_rmse = 0

from sklearn.model_selection import KFold
Kfolds = KFold(n_splits=kfold_splits, shuffle=False)

from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(X_train,y_train)

y_pred=lin.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test,y_pred))

print("Fold" ,num_fold, "xvalid rmse:", rmse)
num_fold = num_fold + 1
oof_rmse += rmse

predictions_linear_regression_test += lin.predict(dataset1[X_train.columns])/kfold_splits

predictions_linear_regression_test = np.expm1(predictions_linear_regression_test)
print()
print(predictions_linear_regression_test)
print()
print("OOF Out-of-fold rmse:", oof_rmse/kfold_splits)


#score prediction

lin.score(X_train,y_train)

lin.score(X_test,y_test)


#                        Decision Tree Regressor

kfold_splits = 5

predictions_decision_tree_regressor_test = np.zeros(len(dataset))
num_fold = 0
oof_rmse = 0

from sklearn.model_selection import KFold
Kfolds = KFold(n_splits=kfold_splits, shuffle=False)

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)

y_pred=lin.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test,y_pred))

print("Fold" ,num_fold, "xvalid rmse:", rmse)
num_fold = num_fold + 1
oof_rmse += rmse

predictions_decision_tree_regressor_test += lin.predict(dataset1[X_train.columns])/kfold_splits

predictions_decision_tree_regressor_test = np.expm1(predictions_decision_tree_regressor_test)
print()
print(predictions_decision_tree_regressor_test)
print()
print("OOF Out-of-fold rmse:", oof_rmse/kfold_splits)

#score prediction

dtr.score(X_train,y_train)

dtr.score(X_test,y_test)



#                            Random Forest

kfold_splits = 5

predictions_random_forest_test = np.zeros(len(dataset))
num_fold = 0
oof_rmse = 0

from sklearn.model_selection import KFold
Kfolds = KFold(n_splits=kfold_splits, shuffle=False)

from sklearn.ensemble import RandomForestRegressor
ran=RandomForestRegressor()
ran.fit(X_train,y_train)

y_pred=ran.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test,y_pred))

print("Fold" ,num_fold, "xvalid rmse:", rmse)
num_fold = num_fold + 1
oof_rmse += rmse


predictions_random_forest_test += ran.predict(dataset1[X_train.columns])/kfold_splits

predictions_random_forest_test = np.expm1(predictions_random_forest_test)
print()
print(predictions_random_forest_test)
print()
print("OOF Out-of-fold rmse:", oof_rmse/num_of_splits)


#score prediction

ran.score(X_train,y_train)

ran.score(X_test,y_test)
