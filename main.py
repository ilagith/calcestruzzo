from src.data_processing import load_data, convert_column_names, feature_target_split, impute_na_and_rescale, \
    extract_features
from src import eda
from src.modelling import get_models_results, define_models, define_params_space, get_feature_importance
from sklearn.model_selection import train_test_split


def main(extract_new_features: True) -> None:
    """
    Main function to:
    - Load data
    - Perform EDA
    - Preprocessing
    - Modelling
    :param extract_new_features: Whether to extract new features or not
    :return:
    """
    df = load_data('dataset.csv')
    df = convert_column_names(df)

    # EDA
    summary_stats = eda.data_summary(df)
    eda.plot_pairs(df, diagonal_plot_distribution='kde')
    eda.examine_missing_values(df)
    eda.plot_missing_values_percentages(df)
    eda.check_correlations(df)
    eda.inspect_outliers(df)
    eda.investigate_duplicates(df)

    # Processing
    if extract_new_features:
        df = extract_features(df, 'both')
    print(df)
    X, y = feature_target_split(df, 'strength')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.8, random_state=123)
    X_train, X_test = impute_na_and_rescale(training_features=X_train, test_features=X_test)

    # Modelling
    models = define_models()
    params_space = define_params_space()
    results = get_models_results(models, params_space, X_train, X_test, y_train, y_test)
    best_performing_model = min(results, key=results.get)
    rmse_best_model = results.get(best_performing_model)[0]
    best_model = results.get(best_performing_model)[1]
    print('baseline', results.get('linear_regression'), 'best_model', best_performing_model, 'rmse', rmse_best_model)

    # Show feature importance
    get_feature_importance(best_performing_model, best_model)


if __name__ == "__main__":
    main()
