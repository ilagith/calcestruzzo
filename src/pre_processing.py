from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt, font_manager as fm
from missingno import matrix as missing_vals_distr
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, RANSACRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from math import sqrt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam
from sklearn.inspection import PartialDependenceDisplay

import pandas as pd
import re
import seaborn as sns
import miceforest as mf
from miceforest import mean_match_default
from scikeras.wrappers import KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

from sklearn.ensemble import GradientBoostingRegressor

pd.set_option("display.max_columns", None)
from scipy.stats import iqr

df = pd.read_csv(filepath_or_buffer='../dataset.csv', sep=';', usecols=lambda col: col != "Unnamed: 0", index_col=0)
df.columns = [re.sub(pattern=r'(?<!^)(?=[A-Z])', repl='_', string=column).lower() for column in df.columns]
# df['clipped_age'] = df.age_in_days.clip(None, 120)
# print(df.columns)
# def cap_outliers_iqr(column, multiplier=1.5):
#     q1 = column.quantile(0.25)
#     print(q1)
#     q3 = column.quantile(0.75)
#     iqr = q3 - q1
#     lower_bound = q1 - multiplier * iqr
#
#     upper_bound = q3 + multiplier * iqr
#     print(upper_bound)
#
#
#     return np.where(column < lower_bound, lower_bound, np.where(column > upper_bound, upper_bound, column))
#
#
# for col in df.columns.values.tolist():
#     df[col] = cap_outliers_iqr(df[col])

plt.hist(df['age_in_days'])
plt.show()
print(df['age_in_days'].unique())
print(df[df['age_in_days']==56.0])
# print(df.info())
# print(df.describe())
# print(df.corr())
#
#
# # Count duplicates based on all columns
# df['size'] = df.groupby(
#     ['cement_component', 'coarse_aggregate_component', 'fine_aggregate_component', 'fly_ash_component', 'age_in_days',
#      'blast_furnace_slag', 'superplasticizer_component', 'water_component'
#      ], as_index=False, dropna=False
# ).transform('size')
#
# to_plot = df.groupby('size', as_index=False).agg(count=('size', 'count'))
# print(to_plot)
# print("TEST", df)
#
# print(df[df['size'] > 1])
# print(df[df['size'] == 11].sort_values(['cement_component', 'blast_furnace_slag']))
# tot = df[df['size'] > 1]['size'].sum()
# print(tot)
# plt.bar(to_plot[to_plot['size'] > 1]['size'], to_plot[to_plot['size'] > 1]['count'])
#
# # Set labels and title
# plt.xlabel('Duplicates')
# plt.ylabel('Count')
# plt.title('Duplicate Records Count')
#
# # Show the plot
# plt.show()
# print(df[(df['size'] == 16)])
# plt.scatter(df[(df['size'] == 14) | (df['size'] == 15) | (df['size'] == 16)]['size'],
#             df[(df['size'] == 14) | (df['size'] == 15) | (df['size'] == 16)]['strength'])
# plt.xlabel('Duplicates')
# plt.ylabel('Strength distribution')
# plt.title('Strength distribution in duplicated rows')
# plt.show()
#
# print(df[['cement_component', 'coarse_aggregate_component', 'fine_aggregate_component', 'fly_ash_component', 'age_in_days', 'blast_furnace_slag', 'superplasticizer_component', 'water_component']].round(1).duplicated(keep=False))
#
#
# df = df.groupby(
#     ['cement_component', 'coarse_aggregate_component', 'fine_aggregate_component', 'fly_ash_component', 'age_in_days',
#      'blast_furnace_slag', 'superplasticizer_component', 'water_component', #'clipped_age'
#      ], as_index=False, dropna=False
# ).agg(new_strength=('strength', iqr))


# Add Features
# df['use_fly_ash_component'] = np.where('fly_ash_component' == 0.0, False, True)
# df['use_superplasticizer_component'] = np.where('superplasticizer_component' == 0.0, False, True)
# df['use_blast_furnace_slag'] = np.where('blast_furnace_slag' == 0.0, False, True)
df['w_c'] = df['water_component'] / df['cement_component']
plt.hist(df['w_c'])
plt.show()

df['cat_hash'] = df['fly_ash_component'] / df['water_component'] #spec gravity
plt.hist(df['cat_hash'])
plt.show()

df['age'] = df['coarse_aggregate_component'] / df['age_in_days'] #spec gravity
plt.hist(df['cat_hash'])
plt.show()

df['age_f'] = df['fine_aggregate_component'] / df['age_in_days'] #spec gravity
#df['age_v'] = df['water_component'] / df['age_in_days'] #spec gravity

# plt.hist(df['cat_hash'])
# plt.show()

# df['f_c'] = df['fly_ash_component'] / df['cement_component'] #spec gravity
# plt.hist(df['cat_hash'])
# plt.show()

# df['w_d'] = df['cat_hash'] / df['w_c'] #more w more days
# plt.hist(df['cat_hash'])
# plt.show()

#np.where(df['fly_ash_component'] < 50.0, True, False)
                          #np.where(df['age_in_days'] >= 200, 'old', 'medium'))

#df = pd.get_dummies(df, columns=['cat_days'])
# print(df)


#
# print(df)
# # pct
# pct = df.isna().mean().round(4) * 100
# print(pct)
# f, ax = plt.subplots()
#
# for i, item in enumerate(zip(pct.keys(), pct.values)):
#     item = list(item)
#     item[1] = item[1].round(2)
#     ax.bar(item[0], item[1], label=item[0])
#     ax.text(i - 0.25, item[1] + 1.50, str(item[1]))
#
#
# ax.set_xticklabels([])
# ax.set_xticks([])
# plt.ylim(0, 100)
# plt.ylabel('NA percentage')
# plt.xlabel('Variables')
# plt.title('% of missing values per variable', size=14, fontweight="bold")
# plt.legend()
# plt.show()
# print("hi")


# Duplicates
# print(df.round(1))
# print(df[df.round(1).duplicated()])
# print(set(df.columns.values.tolist())-set(df['strength']))
# print(df.eval(list(set(df.columns.values.tolist())-set(df['strength']))))
# print(df[df[['cement_component', 'coarse_aggregate_component', 'fine_aggregate_component', 'fly_ash_component', 'age_in_days', 'blast_furnace_slag', 'superplasticizer_component', 'water_component']].round(1).duplicated(keep='first')])
#
# print(df[df[['cement_component', 'coarse_aggregate_component', 'fine_aggregate_component', 'fly_ash_component', 'age_in_days', 'blast_furnace_slag', 'superplasticizer_component', 'water_component']].round(1).duplicated(keep='first')])

# # Create the default pairplot
sns.pairplot(df, diag_kind='hist', corner=True)
plt.show()
#
sns.set(style="darkgrid")
plt.figure(figsize=(15, 5))
sns.boxplot(data=df, orient="h", palette="Set2", dodge=False)
plt.show()
#
# # define the mask to set the values in the upper triangle to True
#
# fig = plt.figure(figsize=(30, 20))
# mask = np.triu(np.ones_like(df.corr(), dtype=bool))
# heatmap = sns.heatmap(df.corr(), annot=True, linewidth=.50, cmap=sns.color_palette("viridis"), mask=mask)
# plt.show()
#
# fig = missing_vals_distr(df, figsize=(25, 25))
# fig = fig.get_figure()
# fig.show()
#

# kernel.plot_imputed_distributions(wspace=0.6, hspace=0.4, datasets=1, iteration=1)
# kernel.plot_imputed_distributions(wspace=0.6, hspace=0.4, datasets=1, iteration=2)
# kernel.plot_imputed_distributions(wspace=0.6, hspace=0.4, datasets=1, iteration=3)
# kernel.plot_imputed_distributions(wspace=0.6, hspace=0.4, datasets=1, iteration=4)
# kernel.plot_imputed_distributions(wspace=0.6, hspace=0.4, datasets=1, iteration=5)
#
# df_test = df.copy(deep=True)
# # df_test.loc[:, :] = IterativeImputer(estimator=RandomForestRegressor()).fit_transform(df)
# # print(df_test)
#
# #df_test = kds.complete_data()
# # Create the default pairplot
#
#
#
#
#
# ### Repeat
# sns.pairplot(df_test, diag_kind='kde')
# plt.show()
#
# sns.set(style="darkgrid")
# sns.set(style="darkgrid")
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=df_test, orient="h", palette="Set2", dodge=False)
# plt.show()
# plt.show()
#
# # define the mask to set the values in the upper triangle to True
#
# fig = plt.figure(figsize=(30, 20))
# mask = np.triu(np.ones_like(df.corr(), dtype=bool))
# heatmap = sns.heatmap(df_test.corr(), annot=True, linewidth=.50, cmap=sns.color_palette("viridis"), mask=mask)
# plt.show()
#
# fig = missing_vals_distr(df_test, figsize=(25, 25))
# fig = fig.get_figure()
# fig.show()
#
# print('hi')
#

##############


# def cap_outliers(data, multiplier=1.5):
#     q1 = np.percentile(data, 25)
#     q3 = np.percentile(data, 75)
#     iqr = q3 - q1
#     lower_bound = q1 - multiplier * iqr
#     upper_bound = q3 + multiplier * iqr
#     capped_data = np.clip(data, lower_bound, upper_bound)
#     return capped_data
#
#
# df = df.apply(cap_outliers)

X = df.loc[:, df.columns != 'strength']
print(X)
y = df['strength']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.8, random_state=123)
print("ORIGINAL TRAIN", X_train)
print("ORIGINAL TEST", X_test)
# kernel = mf.ImputationKernel(
#   X_train,
#   save_all_iterations=True,
#   datasets=1,
#   random_state=42,
#   mean_match_scheme=mean_match_default.set_mean_match_candidates(5)
# )
#
# # print(kernel)
# optimal_parameters, losses = kernel.tune_parameters(
#   dataset=0,
#   verbose=True
# )
# # Run the MICE algorithm for 5 iterations
# kernel.mice(verbose=True, iterations=5, variable_parameters=optimal_parameters)
# X_train = kernel.complete_data()
#
# print(X_train)
#
# kernel = mf.ImputationKernel(
#   X_test,
#   save_all_iterations=True,
#   datasets=1,
#   random_state=42,
#   mean_match_scheme=mean_match_default.set_mean_match_candidates(5)
# )
#
# # print(kernel)
# optimal_parameters, losses = kernel.tune_parameters(
#   dataset=0,
#   verbose=True
# )
# # Run the MICE algorithm for 5 iterations
# kernel.mice(verbose=True, iterations=5, variable_parameters=optimal_parameters)
# X_test = kernel.complete_data()
# print(X_test)

####

pipe_kernel = mf.ImputationKernel(X_train, datasets=1, mean_match_scheme=mean_match_default.set_mean_match_candidates(2)
                                  , random_state=42)

# Define our pipeline
pipe = Pipeline([
    ('impute', pipe_kernel),
    ('scaler', StandardScaler()),
])


# X_train = np.load('X_train.npy')
# X_test = np.load('X_test.npy')
# y_train = np.load('y_train.npy')
# y_test = np.load('y_test.npy')
#
# ####
# transformer = RobustScaler().fit(X_train)
# X_train = transformer.transform(X_train)
# X_test = transformer.transform(X_test)
#####
# # Fit on and transform our training data.
# # Only use 2 iterations of mice.
X_train = pipe.fit_transform(
    X_train,
    #y_train,
    impute__iterations=8
)

scaler = pipe.named_steps['scaler']
#print(scaler.transform([[50.0]]))
# Inverse transform the scaled value to get back the original value
# x_originals = scaler.inverse_transform(X_train[:, -1])
# print(x_originals)
# print(X_train)
# print(X_train[:, -1])
# condition = X_train[:, -1] < 1.26530612  # Example condition, can be any boolean expression\
# cond2 = X_train[:, -1] > 1.66938776
# new_col = np.where(condition, 1, 0)
# new_col2 = np.where(cond2, 1, 0)
# new_col3 = np.where((~condition) & (~cond2), 1, 0)
# # Add the new column to the original array
# X_train = np.column_stack((X_train, new_col))
# X_train = np.column_stack((X_train, new_col2))
# X_train = np.column_stack((X_train, new_col3))
#
# print(X_train)
# Transform the test data as well
X_test = pipe.transform(X_test)
print(X_test.shape)
# condition = X_test[:, -1] < 1.26530612  # Example condition, can be any boolean expression
# cond2 = X_test[:, -1] > 1.66938776
# new_col = np.where(condition, 1, 0)
# new_col2 = np.where(cond2, 1, 0)
# new_col3 = np.where((~condition) & (~cond2), 1, 0)
# # Add the new column to the original array
# X_test = np.column_stack((X_test, new_col))
# X_test = np.column_stack((X_test, new_col2))
# X_test = np.column_stack((X_test, new_col3))
#
# print(X_test)
####
# np.save('X_train.npy', X_train)    # .npy extension is added if not given
# np.save('X_test.npy', X_test)    # .npy extension is added if not given
# np.save('y_train.npy', y_train)    # .npy extension is added if not given
# np.save('y_test.npy', y_test)    # .npy extension is added if not given
###
print("SCALED TRAIN", X_train.shape)
regr = LinearRegression().fit(X_train, y_train)
y_pred = regr.predict(X_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred, squared=False))
#
#
# rr = Ridge().fit(X_train, y_train)
# y_pred = rr.predict(X_test)
# print(r2_score(y_test, y_pred))
# print(mean_squared_error(y_test, y_pred, squared=False))

r = RandomForestRegressor(random_state=123, n_jobs=-1)
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=30, random_state=1).fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred, squared=False))











import tensorflow as tf

import keras_tuner
from tensorflow import keras
from sklearn.model_selection import cross_val_score

def build_model(hp):
  model = keras.Sequential()
  #model.add(tf.keras.Input(shape=(12,)))
  model.add(keras.layers.Dense(
      hp.Choice('units', [8, 16, 32]),
      activation='relu', input_shape=(12,)))
  model.add(keras.layers.Dense(1, activation='linear'))
  hp_learning_rate = hp.Choice('learning_rate', values=list(np.arange(0.001, 0.1, 0.25)))
  model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate))
  return model


tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5)

n_samples = X_train.shape[0]
fold_size = n_samples // 10
indices = np.random.permutation(n_samples)
scores = []
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

for fold in range(10):
    val_indices = np.concatenate((indices[fold * fold_size: (fold + 1) * fold_size], indices[(fold + 1) * fold_size:]))
    train_indices = np.concatenate((indices[:fold * fold_size], indices[(fold + 1) * fold_size:]))
    print(val_indices[:20])
    print(train_indices[:20])
    print(X_train.shape)
    print(y_train.shape)
    X_tr, y_tr = X_train[train_indices], y_train.to_numpy()[train_indices]
    print("TRAIN", X_tr.shape)
    print(y_tr.shape)

    X_val, y_val = X_train[val_indices], y_train.to_numpy()[val_indices]
    print(X_val.shape)
    print(y_val.shape)
    tuner.search(X_tr, y_tr, epochs=50, validation_data=(X_val, y_val), callbacks=[stop_early])

best_model = tuner.get_best_models()[0]
print(best_model)
print(X_test.shape)
# Once trained, you can use the best model to make predictions on new data
predictions = best_model.predict(X_test)

# Calculate the mean squared error on the test data
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", np.sqrt(mse))





#to_plot = pd.DataFrame(X_train, columns=X.columns.values.tolist())

# fig, axs = plt.subplots(2, 4, figsize=(12, 5))
# plt.suptitle('Partial Dependence', y=1.0)
#
# PartialDependenceDisplay.from_estimator(rf, to_plot,
#                                         features=to_plot.columns,
#                                         pd_line_kw={"color": "red"},
#                                         ice_lines_kw={"color": "blue"},
#                                         kind='both',
#                                         ax=axs.ravel()[:len(X_train)])
# plt.tight_layout(h_pad=0.3, w_pad=0.5)
# plt.show()

# # Create and fit the Lasso regression model
# lasso = LassoCV(cv=5)  # Set the regularization parameter alpha
# lasso.fit(X_train, y_train)
#
# # Make predictions on the test set
# y_pred = lasso.predict(X_test)
# #
# # # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)
#
# print("Root Mean Squared Error (RMSE):", rmse)
# print("R-squared (R2) Score:", r2)
#
# def create_model(hidden_units=64, learning_rate=0.001):
#     model = Sequential()
#     model.add(Dense(hidden_units, activation='relu', input_shape=(8,)))
#     model.add(Dense(1, activation='linear'))
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(loss='mean_squared_error', optimizer=optimizer)
#     return model
#
# #model = KerasRegressor(build_fn=create_model, verbose=0)
#
#
#Define the hyperparameters for random search
# param_grid = {
#               'max_depth': list(range(5, 7)),
#               'min_samples_leaf': list(range(30, 51)),
#               'min_samples_split': list(range(2, 20)),
#               'n_estimators': list(range(50, 900, 150)),
#             }
#
# def rmse_scorer(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     return rmse
#
# # Perform cross-validation with RMSE scoring
# random_search = RandomizedSearchCV(r, param_distributions=param_grid, cv=10, scoring='neg_mean_squared_error', n_iter=50, random_state=123)
# random_search.fit(X_train, y_train)
# #
# # cv_results = random_search.cv_results_
# # for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
# #     print("Mean Score:", np.sqrt(-mean_score))
# #     print("Parameters:", params)
# #     print("---")
# # #
# # Get the best model and its parameters
# best_model = random_search.best_estimator_
# best_params = random_search.best_params_
#
# y_pred = best_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
#
# print("RF Best Model Parameters:", best_params)
# print("RF Root Mean Squared Error (RMSE):", rmse)
# #
# # Define the XGBoost model
# model = GradientBoostingRegressor(random_state=1)
# #
# # Define the hyperparameters for random search
# param_grid = {
#     'max_depth': list(range(1, 3)),
#     'n_estimators': list(range(50, 950, 50))
#     #'max_features': ['sqrt', None]
#     #'min_samples_leaf': list(range(1, 10)),
# }
#
# # Define the scoring function for RMSE
# def rmse(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     return rmse
#
# # Create the RandomizedSearchCV object
# random_search = RandomizedSearchCV(model, param_grid, cv=10, scoring='neg_mean_squared_error', random_state=123, n_iter=50)
# random_search.fit(X_train, y_train)
#
# cv_results = random_search.cv_results_
# for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
#     print("Mean Score:", np.sqrt(-mean_score))
#     print("Parameters:", params)
#     print("---")
#
# # Get the best model and its parameters
# best_model = random_search.best_estimator_
# best_params = random_search.best_params_
#
# # Evaluate the best model on the test set
# y_pred = best_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
#
# print("GB Best Model Parameters:", best_params)
# print("GB Root Mean Squared Error (RMSE):", rmse)
# #
# # # # Create an SVM regressor
# # # regressor = SVR(kernel='poly')
# # #
# # # # Train the SVM regressor
# # # regressor.fit(X_train, y_train)
# # #
# # # # Make predictions on the test set
# # # y_pred = regressor.predict(X_test)
# # #
# # # # Evaluate the model
# # # mse = mean_squared_error(y_test, y_pred)
# # # rmse = np.sqrt(mse)
# # # r2 = r2_score(y_test, y_pred)
# # #
# # # print("Mean Squared Error (MSE):", rmse)
# # # print("R-squared (R2) Score:", r2)
# # #
# # # # Create an SVM regressor
# # # regressor = SVR()
# # #
# # # # Train the SVM regressor
# # # regressor.fit(X_train, y_train)
# # #
# # # # Make predictions on the test set
# # # y_pred = regressor.predict(X_test)
# # #
# # # # Evaluate the model
# # # mse = mean_squared_error(y_test, y_pred)
# # # rmse = np.sqrt(mse)
# # # r2 = r2_score(y_test, y_pred)
# # #
# # # print("SVM Mean Squared Error (MSE):", rmse)
# # # print("R-squared (R2) Score:", r2)
# # #
# # # rf = RandomForestRegressor().fit(X_train, y_train)
# # # y_pred = rf.predict(X_test)
# # # print(r2_score(y_test, y_pred))
# # # print("RF",mean_squared_error(y_test, y_pred, squared=False))
# # #
# # # rf = xgb.XGBRegressor().fit(X_train, y_train)
# # # y_pred = rf.predict(X_test)
# # # print(r2_score(y_test, y_pred))
# # # print("XGB", mean_squared_error(y_test, y_pred, squared=False))
# # #
# # # # Create a LightGBM regressor
# # # regressor = lgb.LGBMRegressor()
# # #
# # # # Fit the model to the training data
# # # regressor.fit(X_train, y_train)
# # #
# # # # Make predictions on the test set
# # # y_pred = regressor.predict(X_test)
# # #
# # # # Evaluate the model
# # # r2 = r2_score(y_test, y_pred)
# # #
# # # print("Light Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred, squared=False))
# # # print("R-squared (R2) Score:", r2)
# # #
# #
# # Create a GradientBoostingRegressor
# # regressor = GradientBoostingRegressor()
# #
# # # Fit the model to the training data
# # regressor.fit(X_train, y_train)
# #
# # # Make predictions on the test set
# # y_pred = regressor.predict(X_test)
# #
# # # Evaluate the model
# # mse = mean_squared_error(y_test, y_pred, squared=False)
# # r2 = r2_score(y_test, y_pred)
# #
# # print("GBM Mean Squared Error (MSE):", mse)
# # print("R-squared (R2) Score:", r2)
# #
# model = XGBRegressor(random_state=1) #14.16
# # #
# # # Define the hyperparameters for random search
# param_grid = {
#     'learning_rate': list(np.arange(0.1, 0.3, 0.01)),
#     'max_depth': list(range(1, 3)),
#     'n_estimators': list(range(50, 450, 50)),
#     'lambda': list(np.arange(1, 5.5, 0.25)),
#     'alpha': list(np.arange(0, 2.5, 0.25)),
#     #'min_samples_leaf': list(range(1, 10)),
# }
#
# # Define the scoring function for RMSE
# def rmse(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     return rmse
#
# # Create the RandomizedSearchCV object
# random_search = RandomizedSearchCV(model, param_grid, cv=10, scoring='neg_mean_squared_error', random_state=123, n_iter=50)
# random_search.fit(X_train, y_train)
# #
# # cv_results = random_search.cv_results_
# # for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
# #     print("Mean Score:", np.sqrt(-mean_score))
# #     print("Parameters:", params)
# #     print("---")
# #
# # Get the best model and its parameters
# best_model = random_search.best_estimator_
# best_params = random_search.best_params_
#
# # Evaluate the best model on the test set
# y_pred = best_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
#
# print("XGB Best Model Parameters:", best_params)
# print("XGB Root Mean Squared Error (RMSE):", rmse)
#
# model = lgb.LGBMRegressor(random_state=123, n_jobs=-1)
# # #
# # # Define the hyperparameters for random search
# param_grid = {
#     'num_leaves': list(range(5, 20, 2)),
#     'learning_rate': list(np.arange(0.01, 0.2, 0.01)),
#     'n_estimators': list(range(5, 160, 3)),
#     'reg_lambda': list(np.arange(1, 3, 0.25)),
#     #'min_samples_leaf': list(range(1, 10)),
# }
#
# # Define the scoring function for RMSE
# def rmse(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     return rmse
# #
# # # Create the RandomizedSearchCV object
# random_search = RandomizedSearchCV(model, param_grid, cv=10, scoring='neg_mean_squared_error', random_state=123, n_iter=50)
# random_search.fit(X_train, y_train)
# #
# # cv_results = random_search.cv_results_
# # for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
# #     print("Mean Score:", np.sqrt(-mean_score))
# #     print("Parameters:", params)
# #     print("---")
# #
# # Get the best model and its parameters
# best_model = random_search.best_estimator_
# best_params = random_search.best_params_
#
# # Evaluate the best model on the test set
# y_pred = best_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
#
# print("LGBM Best Model Parameters:", best_params)
# print("LGBM Root Mean Squared Error (RMSE):", rmse)
#
# # Get feature importances
# feature_importance = best_model.feature_importances_
#
# # Create bar plot
# plt.bar(range(len(feature_importance)), feature_importance)
# plt.xlabel('Feature Index')
# plt.ylabel('Importance Score')
# plt.title('Feature Importances')
# plt.show()
