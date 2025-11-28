import streamlit as st
import pickle
import pandas as pd

#with open(r'D:\AI ML\ml_projects\customer churn prediction\ride price prediction\one1.pkl', 'rb') as f:
    #one = pickle.load(f)
    
#one1 = pickle.load(open("encoder.pkl", "rb"))    

with open("encoder.pkl", 'rb') as f:
    one1 = pickle.load(f)
    
with open("le1.pkl", 'rb') as f:
    le = pickle.load(f) 
    
with open("linear_model1.pkl", 'rb') as f:
    linear=pickle.load(f)  
    
with open("feature_order.pkl", 'rb') as f:
    feature_order=pickle.load(f)  
    
with open("ride_catboost.pkl",'rb') as f:
    catboost=pickle.load(f) 
    
with open("stacked_model_ride.pkl",'rb') as f:
    stacked_model=pickle.load(f)      
    
from sklearn.metrics import accuracy_score 


       

st.set_page_config(page_title="Ride Price Prediction", page_icon="🚕", layout="centered")

st.title("Ride Price Prediction App 🚕")


#Number_of_Riders	Number_of_Drivers	Location_Category	Customer_Loyalty_Status	Number_of_Past_Rides	Average_Ratings	Time_of_Booking	Vehicle_Type	Expected_Ride_Duration
Number_of_Riders = st.number_input("Number of Riders", min_value=1, max_value=100, value=1)
Number_of_Drivers = st.number_input("Number of Drivers", min_value=1, max_value=100, value=1)
Number_of_Past_Rides= st.number_input("Number of Past Rides", min_value=0, max_value=1000, value=0)
Average_Ratings= st.number_input("Average Ratings", min_value=1.0, max_value=5.0, value=5.0)
Vehicle_Type = st.selectbox("Vehicle Type", le.classes_)
Expected_Ride_Duration= st.number_input("Expected Ride Duration (in minutes)", min_value=1, max_value=300, value=10)
Location_Category= st.selectbox("Location Category", one1.categories_[1])
Customer_Loyalty_Status= st.selectbox("Customer Loyalty Status", one1.categories_[0])
Time_of_Booking= st.selectbox("Time of Booking", one1.categories_[2])


vehicle_map = {"Economy": 0, "Premium": 1}
input_data=pd.DataFrame({
    'Number_of_Riders':[Number_of_Riders],  
    'Number_of_Drivers':[Number_of_Drivers],
    'Number_of_Past_Rides':[Number_of_Past_Rides],
    'Average_Ratings':[Average_Ratings],
    'Vehicle_Type':[vehicle_map[Vehicle_Type]],
    'Expected_Ride_Duration':[Expected_Ride_Duration],
    
   })

input_df=pd.DataFrame({
    'Customer_Loyalty_Status':[Customer_Loyalty_Status],
    'Location_Category':[Location_Category],
    'Time_of_Booking':[Time_of_Booking]
    
})


#if st.button("Predict Ride Price using linear", key="predict_linear"):
    # linear prediction code here
 #   pass

#if st.button("Predict Ride Price using Catboost", key="predict_catboost"):
    # catboost prediction code here
 #   pass



# Encode categorical columns
encoded = one1.transform(input_df[['Customer_Loyalty_Status','Location_Category', 'Time_of_Booking']])

# Correct way to get output column names
encoded_cols = one1.get_feature_names_out()

encoded_df = pd.DataFrame(encoded.toarray(), columns=encoded_cols)







#encoded_cols = one1.get_feature_names_out(['Location_Category', 
 #                                          'Customer_Loyalty_Status',
  #                                         'Time_of_Booking'])

#encoded_df = pd.DataFrame(one1.transform(input_df).toarray(), 
 #                         columns=encoded_cols)

final = pd.concat([input_data.reset_index(drop=True), 
                   encoded_df.reset_index(drop=True)], axis=1)

final=final.reindex(columns=feature_order, fill_value=0)


#encoded_df = one1.transform(input_df).toarray()

#final=pd.concat([input_data, pd.DataFrame(encoded_df)], axis=1, ignore_index=True)


if st.button("Predict Ride Price using linear"):
    
    vehicle_map = {"Economy": 0, "Premium": 1}
    input_data=pd.DataFrame({
    'Number_of_Riders':[Number_of_Riders],  
    'Number_of_Drivers':[Number_of_Drivers],
    'Number_of_Past_Rides':[Number_of_Past_Rides],
    'Average_Ratings':[Average_Ratings],
    'Vehicle_Type':[vehicle_map[Vehicle_Type]],
    'Expected_Ride_Duration':[Expected_Ride_Duration],
    
    })
    
    input_df=pd.DataFrame({
    'Customer_Loyalty_Status':[Customer_Loyalty_Status],
    'Location_Category':[Location_Category],
    'Time_of_Booking':[Time_of_Booking]
    
    })
    
    # Encode categorical columns
    encoded = one1.transform(input_df[['Customer_Loyalty_Status','Location_Category', 'Time_of_Booking']])

# Correct way to get output column names
    encoded_cols = one1.get_feature_names_out()

    encoded_df = pd.DataFrame(encoded.toarray(), columns=encoded_cols)
    final = pd.concat([input_data.reset_index(drop=True), 
                   encoded_df.reset_index(drop=True)], axis=1)

    final=final.reindex(columns=feature_order, fill_value=0)
    linear_prediction=linear.predict(final)
    st.success(f"The Price of the ride is Rs.{linear_prediction}")
elif st.button("Predict Ride Price using Catboost"):
     vehicle_map = {"Economy": 0, "Premium": 1}
     input_data=pd.DataFrame({
    'Number_of_Riders':[Number_of_Riders],  
    'Number_of_Drivers':[Number_of_Drivers],
    'Number_of_Past_Rides':[Number_of_Past_Rides],
    'Average_Ratings':[Average_Ratings],
    'Vehicle_Type':[vehicle_map[Vehicle_Type]],
    'Expected_Ride_Duration':[Expected_Ride_Duration],
    
    })
     input_df=pd.DataFrame({
    'Customer_Loyalty_Status':[Customer_Loyalty_Status],
    'Location_Category':[Location_Category],
    'Time_of_Booking':[Time_of_Booking]
    
    })
     # Encode categorical columns
     encoded = one1.transform(input_df[['Customer_Loyalty_Status','Location_Category', 'Time_of_Booking']])

# Correct way to get output column names
     encoded_cols = one1.get_feature_names_out()

     encoded_df = pd.DataFrame(encoded.toarray(), columns=encoded_cols)
     final = pd.concat([input_data.reset_index(drop=True), 
                   encoded_df.reset_index(drop=True)], axis=1)

     final=final.reindex(columns=feature_order, fill_value=0)
     catboost_prediction=catboost.predict(final)
     st.success(f"The Price of the ride is Rs.{catboost_prediction}")
elif st.button("Predict Ride Price using Stack Model"):
     vehicle_map = {"Economy": 0, "Premium": 1}
     input_data=pd.DataFrame({
    'Number_of_Riders':[Number_of_Riders],  
    'Number_of_Drivers':[Number_of_Drivers],
    'Number_of_Past_Rides':[Number_of_Past_Rides],
    'Average_Ratings':[Average_Ratings],
    'Vehicle_Type':[vehicle_map[Vehicle_Type]],
    'Expected_Ride_Duration':[Expected_Ride_Duration],
    
    })
     input_df=pd.DataFrame({
    'Customer_Loyalty_Status':[Customer_Loyalty_Status],
    'Location_Category':[Location_Category],
    'Time_of_Booking':[Time_of_Booking]
    
    })
     # Encode categorical columns
     encoded = one1.transform(input_df[['Customer_Loyalty_Status','Location_Category', 'Time_of_Booking']])

# Correct way to get output column names
     encoded_cols = one1.get_feature_names_out()

     encoded_df = pd.DataFrame(encoded.toarray(), columns=encoded_cols)
     final = pd.concat([input_data.reset_index(drop=True), 
                   encoded_df.reset_index(drop=True)], axis=1)

     final=final.reindex(columns=feature_order, fill_value=0)
     stacked_prediction=stacked_model.predict(final)
     st.success(f"The Price of the ride is Rs.{stacked_prediction}")     
    
    








