#importing libraries
import pandas as pd
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.ensemble import RandomForestClassifier

#Title
st.title("Shipments Prediction")
st.sidebar.header("User input variables")

#Input Featurs
def Input_features():
   Call = st.sidebar.number_input("Customer Calls",min_value=0,max_value=10,value=0)
   Ratings = st.sidebar.number_input("Customer Rating",min_value=0,max_value=5,value=0)
   Cost_Of_Product = st.sidebar.number_input("Product Cost",min_value=0,max_value=9999,value=0)
   Purchases = st.sidebar.number_input("Prior Purchases",min_value=0,max_value=9,value=0)
   Discount_offer = st.sidebar.number_input("Discount Offer",min_value=0,max_value=100,value=0,step=1)
   Weight = st.sidebar.number_input("Weight",min_value=0,max_value=9999,value=0,step=50)
   WARE_B = st.sidebar.number_input('Warehouse_B',min_value=0,max_value=1,value=0)
   WARE_C = st.sidebar.number_input('Warehouse_C',min_value=0,max_value=1,value=0)
   WARE_D = st.sidebar.number_input('Warehouse_D',min_value=0,max_value=1,value=0)
   WARE_F = st.sidebar.number_input('Warehouse_F',min_value=0,max_value=1,value=0)
   
#data input 
   data={
          'Call':Call,
          'Ratings':Ratings,
          'Cost_Of_Product':Cost_Of_Product,
          'Purchases':Purchases,
          'Discount_offer':Discount_offer,
          'Weight':Weight,
          'WARE_B':WARE_B,
          'WARE_C':WARE_C,
          'WARE_D':WARE_D,
          'WARE_F':WARE_F}
   features=pd.DataFrame(data,index=[0])
   return features
    
df=Input_features()
st.subheader("Input Parameters")
st.write(df)


#importing data
cust_data= pd.read_csv("/Users/om/Desktop/Project/Shipment/Final_files/shipments.csv")
cust_data.drop('ID',axis=1,inplace=True)
cust_data=cust_data.dropna()

cust_data.rename(columns={'Reached.on.Time_Y.N':'Reached_on_time'}, inplace=True)
cust_data["Reached_on_time"]=cust_data["Reached_on_time"].apply(lambda x: "Yes" if x==0 else "Not")

#converting obj to cat 
cust_data["Warehouse_block"] = cust_data.Warehouse_block.astype('category')
cust_data["Mode_of_Shipment"] = cust_data.Mode_of_Shipment.astype('category')
cust_data["Product_importance"] = cust_data.Product_importance.astype('category')
cust_data["Gender"] = cust_data.Gender.astype('category')
cust_data["Reached_on_time"] = cust_data.Reached_on_time.astype('category')

#copy dataset
cust_data_clean=cust_data.copy()

#outlier removing
Q1_purchase=cust_data.Prior_purchases.quantile(.25)
Q3_purchase = cust_data.Prior_purchases.quantile(0.75)
IQR_purchase=Q3_purchase-Q1_purchase
Out_purchase=cust_data[((cust_data.Prior_purchases<(Q1_purchase-1.5*IQR_purchase)) | (cust_data.Prior_purchases>(Q3_purchase+1.5*IQR_purchase)))]


ninetieth_percentile_purchase = np.percentile(cust_data_clean.Prior_purchases, 90)
cust_data_clean.Prior_purchases = np.where(cust_data_clean.Prior_purchases > ninetieth_percentile_purchase, ninetieth_percentile_purchase, cust_data_clean.Prior_purchases)

#dummies
cust_data_cleaned = pd.get_dummies(cust_data, prefix_sep="_", drop_first=True)
array=cust_data_cleaned.values

#minmax Scaler
scaler=MinMaxScaler()
scaled_data=scaler.fit_transform(array)
scaled_data=pd.DataFrame(scaled_data)

x=scaled_data
y=cust_data_cleaned.iloc[:,-1]

#feature selection
test=SelectKBest(score_func=chi2,k=10)
fit_data=test.fit(x,y)
features=fit_data.transform(x)

#Splitting data
trainX, testX, trainY, testY = train_test_split(features,y,test_size=0.2,random_state=42)

#Random Forest Model
model_RF=RandomForestClassifier(n_estimators=50,max_features=2)
model_RF.fit(features,y)

st.title("Predicted Result")
prediction=model_RF.predict(df)

#Output
if prediction==1:
    pred=('Order will be on time.')
    st.balloons()
else:
    pred=('Order will be delayed.')


if st.button("Predict"):
   st.write(pred)
st.button("Clear Predict")




