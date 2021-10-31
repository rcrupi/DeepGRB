from models.load_data import df_burst_catalog
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    # Load GRB GBM table with selected column and drop NaN values
    df_grb = df_burst_catalog(download=False, dropna=True)
    # Define X input and y target for regression
    X = df_grb.loc[:, [i for i in df_grb.columns if i != 't90']].astype('float64')
    y = np.log(df_grb['t90'])
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    # # Train a random forest
    # rf = RandomForestRegressor(n_estimators=4000, criterion='mse', max_depth=20,
    #                            min_samples_split=10, min_samples_leaf=3, min_weight_fraction_leaf=0.0,
    #                            max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
    #                            min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1,
    #                            random_state=0, verbose=1, warm_start=False, ccp_alpha=0.0, max_samples=None)
    #
    # # rf = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    # #                       colsample_bynode=1, colsample_bytree=1, gamma=0,
    # #                       importance_type='gain', learning_rate=0.1, max_delta_step=0,
    # #                       max_depth=4, min_child_weight=1, missing=None, n_estimators=400,
    # #                       n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,
    # #                       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
    # #                       silent=None, subsample=1, verbosity=1)
    # rf.fit(X_train, y_train)
    # y_pred = rf.predict(X_test)
    #
    # plt.figure(figsize=[10, 6])
    # plt.plot(y_test, y_pred, '.', alpha=0.5)
    # max_y_val = max(y_test); max_y_val = max(max_y_val, max(y_pred))
    # min_y_val = min(y_test); min_y_val = min(min_y_val, min(y_pred))
    # plt.plot([min_y_val, max_y_val], [min_y_val, max_y_val], '-', alpha=0.2)
    # plt.title('Test set')
    #
    # y_pred_train = rf.predict(X_train)
    # plt.figure(figsize=[10, 6])
    # plt.plot(y_train, y_pred_train, '.', alpha=0.5)
    # max_y_val = max(y_train); max_y_val = max(max_y_val, max(y_pred_train))
    # min_y_val = min(y_train); min_y_val = min(min_y_val, min(y_pred_train))
    # plt.plot([min_y_val, max_y_val], [min_y_val, max_y_val], '-', alpha=0.2)
    # plt.title('Train set')
    #
    # import shap  # package used to calculate Shap values
    #
    # # Create object that can calculate shap values
    # explainer = shap.TreeExplainer(rf, X_test)
    # # calculate shap values. This is what we will plot.
    # # Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
    # shap_values = explainer.shap_values(X_test, y_test, check_additivity=False)
    # # Make plot. Index of [1] is explained in text below.
    # plt.figure()
    # shap.summary_plot(shap_values, X_test)
    # # make interaction plot
    # #plt.figure()
    # shap.dependence_plot('flux_1024', shap_values, X_test, interaction_index="flnc_plaw_phtflux")
    #
    # # f, ax = plt.subplots(figsize=(10, 8))
    # # corr = X.corr()
    # # sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
    # #             square=True, ax=ax)
    col_selected_final_rf =['flnc_comp_ergflux',
 'flux_1024',
 'flnc_plaw_phtflux',
 'pflx_band_alpha',
 'pflx_comp_epeak',
 'pflx_plaw_phtflux',
 'flnc_sbpl_pivot',
 'flnc_plaw_pivot',
 'pflx_comp_pivot']# ['flnc_plaw_index', 'flnc_band_ampl', 'pflx_band_phtflux',
                   #          'flux_256', 'flnc_plaw_phtflux', 'flux_64', 'pflx_sbpl_indx1',
                   #          'pflx_comp_phtfluxb', 'flnc_comp_phtflux']
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df_grb[col_selected_final_rf])
    X_tsne = scaler.transform(df_grb[col_selected_final_rf])

    from sklearn.manifold import TSNE
    import matplotlib

    for perp in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]: #32
        X_embedded = TSNE(n_components=2, perplexity=perp, early_exaggeration=12.0,
                          learning_rate=200.0, n_iter=1000, n_iter_without_progress=300,
                          min_grad_norm=1e-07, metric='euclidean', init='random', verbose=1,
                          random_state=32, method='barnes_hut', angle=0.5, n_jobs=-1).fit_transform(X_tsne)
        plt.figure()
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                    c=df_grb.loc[:, 't90'], alpha=1, s=1,  # # df_table['T90']
                    cmap='viridis', norm=matplotlib.colors.LogNorm())
        plt.colorbar()
        plt.legend()
        plt.title('perplexity: ' + str(perp))
    plt.show()

    pass


