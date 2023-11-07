import streamlit as st
import os

import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from IPython.display import display, Markdown
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

import joblib
import lightgbm as lgb

import lxml
#import eli5
#from eli5.sklearn import PermutationImportance
from sklearn.metrics import mean_squared_error
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
from sklearn.inspection import plot_partial_dependence

import warnings
warnings.filterwarnings('ignore') 


# Creating the Layout of the App
#st.set_page_config(layout="wide")

# 数据缓存
@st.cache_data
def load_data(file_name):
    return pd.read_csv(file_name)


def quantile_predictions(X_train, y_train,X_valid):
    predictions=[]

    # 预测在不同分位点下的值
    quantile_levels = [0.1, 0.5, 0.9]  # 选择不同分位点
    for i, q in enumerate(quantile_levels):  
            lgb_params = {
            "objective": "quantile",
            "alpha": q,  # 分位点 (0.1, 0.5, 0.9など)
            "boosting_type": "gbdt",
            "num_leaves": 36,
            "learning_rate": 0.05,
            "n_estimators": 160,
            "max_depth": 20,
            "reg_lambda": 0.8728250152248831,
            "random_seed": 0,
        }
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_train, y_train, eval_set=None)
            # 予測を取得
            quantile_predictions = model.predict(X_valid, num_iteration=model.best_iteration_)
            predictions.append(quantile_predictions)           
    for i, q in enumerate(quantile_levels):
            predic_1=predictions[0]
            predic_5=predictions[1]
            predic_9=predictions[2]                  
            return predic_1,predic_5,predic_9

def get_heatmap(train_df):
       #特徴量と目的変数間の相関係数を計算   
        fig, ax = plt.subplots(figsize=(10,10))   
        data=train_df.drop(columns='Date')
        sns.heatmap(data.corr(),ax=ax,annot=True,fmt='.2f')  #fmt='.2f' 0.01
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        return fig
    
#def calculate_permutation_importance(model, X_valid,y_valid):
#    perm = PermutationImportance(model, random_state=1).fit(X_valid, y_valid)
#    features_weights = eli5.show_weights(perm, top=len(X_valid.columns), feature_names = X_valid.columns.tolist())
#    features_weight = pd.read_html(features_weights.data)[0]
#    return features_weight



def main():
    # Title
    st.title("Demand Forecasting Demo")
    st.markdown(
        """
        We used kaggle walmart dataset as features ,
        and we devided them into three parts,
        training,Validation,test   
        You can check detail from the tab below
        """ )
    
    # 读取数据
    train_data = load_data("walmart_train_modify.csv")
    var_data = load_data("walmart_var_modify.csv")
    no_use_columns=['consecutive_holidays','Date','Weekly_Sales','Shop_Type','Resident_Population','Type',
                        'MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','Year','Dept']
    features = [i for i in train_data.columns if i not in no_use_columns] # List of used columns
    
    X_train = train_data[features].copy()
    y_train = train_data.Weekly_Sales.copy()
    X_valid=var_data[features].copy()
    y_valid = var_data.Weekly_Sales.copy()
    
    
    
    #train model 
    LGBM=lgb.LGBMRegressor(n_estimators= 160, 
                             max_depth=20,
                             num_leaves=36,
                             reg_lambda=0.8728250152248831,
                            random_seed=0,
                           #objective='quantile', alpha=0.4
                            )
    LGBM.fit(X_train,y_train, eval_set=None)
    y_preds_train=  LGBM.predict(X_train)
    y_preds_var=  LGBM.predict(X_valid)

    
            
    # 显示数据
    st.header("Datasets")
    st.markdown(
        """
        Select Dataset
        """ )
    
    # タブ
    tab1, tab2 = st.tabs(["Train_data", "Valid_data"])
    with tab1:
        st.write("Training data set from 2010-03-05 to 2012-02-24")     
        st.dataframe(train_data.head(3).append(train_data.tail(3)))
    with tab2:
        st.write("Validation data set from 2012-03-02 to 2012-10-26")
        st.dataframe(var_data.head(3).append(var_data.tail(3)))
        

   
    # 显示模型的性能
    st.subheader("Model Performance")

    if st.button("Show Model Performance"):
        mse = np.sqrt(mean_squared_error(y_valid, y_preds))
        st.write(f'Mean Squared Error: {mse:.4f}')

    # show Graph
    st.header("Prediction Graph")
    num1=X_train['Store'].unique()
    option1 = st.selectbox('Select a Store',num1)
    
    # show trend
    st.subheader(f'Weekly Sales Trend of Store:{option1}')
    if st.button('Show Trend'):
        
        predic_1,predic_5,predic_9=quantile_predictions(X_train, y_train,X_valid)
        # train_index
        train_data['Date'] = pd.to_datetime(train_data['Date'], format='%Y-%m')
        x=train_data.set_index('Date')
        x['y_preds_train'] = y_preds_train
        x1=x[(x.Store==option1)]
        # var_data_index
        var_data['Date'] = pd.to_datetime(var_data['Date'], format='%Y-%m')
        y=var_data.set_index('Date')
        y['y_preds_var'] = y_preds_var
        y['predic_1'] = predic_1
        #y['predic_5'] =predic_5
        y['predic_9'] = predic_9
        y1=y[(y.Store==option1)]

        figure_trend, ax = plt.subplots(figsize=(20,10))   
        ax.plot(x1.index, x1.Weekly_Sales, label="Actual(train)")
        ax.plot(y1.index, y1.Weekly_Sales, label="Actual(validation")
        ax.plot(x1.index, x1.y_preds_train, label="Predict(train))", linestyle="dotted", lw=2, color="m") 
        ax.plot(y1.index, y1.y_preds_var, label="Predict(validation)", linestyle="dotted", lw=2, color="m") 
        ax.plot(y1.index, y1.predic_1, label="predic_1(validation)", linestyle="dotted", lw=2, color="Black") 
        #ax.plot(y1.index, y1.predic_5, label="predic_5(validation)", linestyle="dotted", lw=2, color="Black") 
        ax.plot(y1.index, y1.predic_9, label="predic_9(validation)", linestyle="dotted", lw=2, color="Black") 
        
        ax.axvline(x=pd.to_datetime('2012-03', format='%Y-%m'), color='red', linestyle='--', label="Vertical Line at x=5")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        #ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.tick_params(labelsize=15)
        
        # enlarge label
        legend = ax.legend()       
        for text in legend.get_texts():
            text.set_fontsize(15)  
                                  
        st.pyplot(figure_trend)
    
    
    
    # show heatmap
    st.header("Heatmap")

    if st.button('Show Heatmap'):
        fig=get_heatmap(train_data)
        st.write(fig)

        
    
    st.header("permutation_importance")   
    if st.button('Show Permutation Importance'):                                      
        #perm_importance = permutation_importance(LGBM, X_train,y_train, n_repeats=10, random_state=71)
        #sorted_idx = perm_importance.importances_mean.argsort()                                                 
        #figure = go.Figure(data=[go.Bar(x=X_train.columns[sorted_idx], y=perm_importance.importances_mean[sorted_idx])])
        #features_weight = calculate_permutation_importance(LGBM, X_valid, y_valid)
        
        #sorted_idx = features_weight['feature'].str.split(' ').str[-1].astype(int).argsort()

        loaded_results = np.load('permutation_importance_results.npy')
        sorted_idx = loaded_results.argsort()                                                 
        figure = go.Figure(data=[go.Bar(x=X_train.columns[sorted_idx], y=loaded_results[sorted_idx])])
        # create graph
        figure, ax = plt.subplots(figsize=(10,10))
        #ax.barh(X_train.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
        ax.barh(X_train.columns[sorted_idx], loaded_results[sorted_idx])
        ax.set_xlabel('Permutation Importance', fontsize=15)
        ax.set_ylabel('Feature', fontsize=15)
        ax.set_title('Permutation Importance', fontsize=15)
        # organize x,y axis label size
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        # Display graph on Streamlit 
        st.pyplot(figure)

                                                         
    
    st.header("Partial Dependence Plots")
    select_feature = st.selectbox('Select a Feature for Partial Dependence Plots', X_train.columns)
    
    if st.button("Calculate Partial Dependence"):    
        st.write(f"You selected: {select_feature}, Calculating {select_feature} partial dependence")

        # Draw graph
        PDP, ax = plt.subplots(figsize=(10,10))
        plot_partial_dependence(LGBM, X_train, features=[select_feature], grid_resolution=50, ax=ax)
        plt.suptitle(f'Partial Dependence Plot for {select_feature}', fontsize=15)
        ax.set_ylabel('Partial Dependence', fontsize=15)
        ax.set_xlabel(f'{select_feature}', fontsize=15)
        # organize x,y axis label size
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        
        plt.subplots_adjust(top=0.9)
    

        # show graph
        st.pyplot(PDP)
    else:
        st.session_state.feature_select = None
        
        
        

    
    
if __name__ == "__main__":
    main()
