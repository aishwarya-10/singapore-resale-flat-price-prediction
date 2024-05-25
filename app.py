# ==================================================       /     IMPORT LIBRARY    /      =================================================== #
#[model]
import pickle
import base64

#[Data Transformation]
import numpy as np
import pandas as pd

#[Dashboard]
import plotly.graph_objects as go
import streamlit as st
from streamlit_extras.stylable_container import stylable_container

#[Map]
import folium
import geopy.geocoders as geocoders


# ==================================================       /     CUSTOMIZATION    /      =================================================== #
# Streamlit Page Configuration
st.set_page_config(
    page_title = "Flat Price Predictor",
    page_icon= "Images/diagram.png",
    layout = "wide",
    initial_sidebar_state= "expanded"
    )

# Title
st.title(":blue[Singapore Resale Flat Price Prediction]")

# Intro
st.write(""" """)


# ==================================================       /     SIDE BAR    /      =================================================== #

# options
flat_type = ['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', 'MULTI GENERATION']
flat_type_dict = {'1 ROOM': 0, '2 ROOM': 1, '3 ROOM': 2, '4 ROOM': 3, '5 ROOM':4, 'EXECUTIVE': 5, 'MULTI GENERATION': 6}

storey_range = ['10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15',
       '19 TO 21', '16 TO 18', '25 TO 27', '22 TO 24', '28 TO 30',
       '31 TO 33', '40 TO 42', '37 TO 39', '34 TO 36', '06 TO 10',
       '01 TO 05', '11 TO 15', '16 TO 20', '21 TO 25', '26 TO 30',
       '36 TO 40', '31 TO 35', '46 TO 48', '43 TO 45', '49 TO 51']
storey_range_dict = {'10 TO 12': 0, '04 TO 06': 1, '07 TO 09': 2, '01 TO 03': 3, '13 TO 15': 4,
       '19 TO 21': 5, '16 TO 18': 6, '25 TO 27': 7, '22 TO 24': 8, '28 TO 30': 9,
       '31 TO 33': 10, '40 TO 42': 11, '37 TO 39': 12, '34 TO 36': 13, '06 TO 10': 14,
       '01 TO 05': 15, '11 TO 15': 16, '16 TO 20': 17, '21 TO 25': 18, '26 TO 30': 19,
       '36 TO 40': 20, '31 TO 35': 21, '46 TO 48': 22, '43 TO 45': 23, '49 TO 51': 24}

flat_model = ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
       'Model A-Maisonette', 'Apartment', 'Maisonette', 'Terrace',
       '2-room', 'Improved-Maisonette', 'Multi Generation',
       'Premium Apartment', 'Adjoined flat', 'Premium Maisonette',
       'Model A2', 'DBSS', 'Type S1', 'Type S2', 'Premium Apartment Loft',
       '3Gen']
flat_model_dict = {'Improved': 0, 'New Generation': 1, 'Model A': 2, 'Standard': 3, 'Simplified': 4,
       'Model A-Maisonette': 5, 'Apartment': 6, 'Maisonette': 7, 'Terrace': 8,
       '2-room': 9, 'Improved-Maisonette': 10, 'Multi Generation': 11,
       'Premium Apartment': 12, 'Adjoined flat': 13, 'Premium Maisonette': 14,
       'Model A2': 15, 'DBSS': 16, 'Type S1': 17, 'Type S2':18, 'Premium Apartment Loft': 19,
       '3Gen': 20}

town = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
       'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI',
       'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
       'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG',
       'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN',
       'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS',
       'PUNGGOL']
town_dict = {'ANG MO KIO': 0, 'BISHAN': 1, 'BUKIT BATOK': 2, 'BUKIT MERAH': 3,
       'BUKIT TIMAH': 4, 'CENTRAL AREA': 5, 'CHOA CHU KANG': 6, 'CLEMENTI': 7,
       'GEYLANG': 8, 'HOUGANG': 9, 'JURONG EAST': 10, 'JURONG WEST': 11,
       'KALLANG/WHAMPOA': 12, 'MARINE PARADE': 13, 'QUEENSTOWN': 14, 'SENGKANG': 15,
       'SERANGOON': 16, 'TAMPINES': 17, 'TOA PAYOH': 18, 'WOODLANDS': 19, 'YISHUN': 20,
       'LIM CHU KANG': 21, 'SEMBAWANG': 22, 'BUKIT PANJANG': 23, 'PASIR RIS': 24,
       'PUNGGOL': 25,  'BEDOK': 26}

with st.sidebar:
    st.header("Apartment Details :cityscape:")

    m_town = st.selectbox(label= "Town", options= town, index= 0, key= "town")
    m_flat_type = st.selectbox(label= "Flat Type", options= flat_type, index= 0, key= "flat_type")
    m_storey_range = st.selectbox(label= "Storey Range", options= storey_range, index= 0, key= "storey")
    m_flat_model = st.selectbox(label= "Flat Model", options= flat_model, index= 0, key= "flat_model")
    m_lease_commence = st.slider(label= "Lease Commence (year)", value= 2017, min_value= 1990, max_value= 2024, key= "commence")
    m_remain_lease = st.slider(label= "Remaining Lease (year)", value= 50, min_value= 0, max_value= 99, key= "re_lease")
    m_floor_area = st.slider(label= "Floor Area (sq.m)", value= 50, min_value= 25, max_value= 310, key= "area")

    with stylable_container(
        key="red_button",
        css_styles="""
            button {
                background-color: green;
                color: white;
                border-radius: 20px;
                background-image: linear-gradient(90deg, #0575e6 0%, #021b79 100%);
            }
            """,
    ):  
        pred_price_button = st.button("Estimate Resale Price")



# ==================================================       /     MODEL    /      =================================================== #

def predict_resale_price():

    # Load pre-trained model
    model = pickle.load(open("Model/resale_price_predictor1.pkl", "rb"))
    scaler = pickle.load(open("Model/scaler.pkl", "rb"))

    # Combine user inputs to an array
    user_data = np.array([[int(town_dict.get(m_town)),
                           int(flat_type_dict.get(m_flat_type)),
                           int(storey_range_dict.get(m_storey_range)),
                           int(flat_model_dict.get(m_flat_model)),
                           int(m_lease_commence),
                           int(m_remain_lease),
                           int(m_floor_area)
                           ]])

    prediction = model.predict(user_data)

    y_p = prediction.reshape(1, -1)
    y_predicted_original = scaler.inverse_transform([[1, y_p]])[0][1]

    return y_predicted_original


if pred_price_button:
    predicted_price = predict_resale_price()
    st.subheader("Price Prediction")
    st.write("The estimated resale price is ", round(predicted_price, 4))


# ==================================================       /     PAST TRANSACTIONS    /      =================================================== #
# Database
st.subheader("Past Transactions :moneybag:")
df = pd.read_csv("Data/pre_processed_data2.csv")
df["address"] = df["block"] + ", " + df["street_name"] + ", Singapore"

filtered_df = df[(df["town"] == str(m_town)) & (df["flat_type"] == str(m_flat_type))]
st.dataframe(filtered_df)


def get_lat_long(df, address_column="address", lat_column="latitude", lon_column="longitude"):
  """
  This function takes a DataFrame and iterates through the specified address column to geocode
  each address and add latitude and longitude columns to the DataFrame.

  Args:
      df (pandas.DataFrame): The DataFrame containing the addresses.
      address_column (str, optional): The name of the column containing addresses. Defaults to "address".
      lat_column (str, optional): The name of the column to store latitude values. Defaults to "latitude".
      lon_column (str, optional): The name of the column to store longitude values. Defaults to "longitude".

  Returns:
      pandas.DataFrame: The modified DataFrame with latitude and longitude columns.
  """

  # Using Nominatim as an example
  geolocator = geocoders.Nominatim(user_agent="myGeocoder")

  # Add latitude and longitude columns if they don't exist
  # if lat_column not in df.columns:
  #   df[lat_column] = None
  # if lon_column not in df.columns:
  #   df[lon_column] = None

  for index, row in df.iterrows():
    address = row[address_column]
    location = geolocator.geocode(address)

    if location is not None:
      latitude = location.latitude
      longitude = location.longitude
      df.at[index, lat_column] = latitude
      df.at[index, lon_column] = longitude
    else:
      df.at[index, lat_column] = None
      df.at[index, lon_column] = None

  return df


# address_df = get_lat_long(df)

# address_df["locations"] = address_df["latitude"] + ", " + address_df["longitude"]

# locations = pd.to_list(df["locations"])

# # Create a base map centered on the first location
# map = folium.Map(location=locations[0], zoom_start=5)


# # Function to create a building icon
# def create_building_icon(color='blue'):
#   """
#   This function creates a custom building icon for markers.

#   Args:
#       color (str, optional): The color of the icon. Defaults to 'blue'.

#   Returns:
#       folium.Icon: The created building icon.
#   """
#   icon_create = folium.Icon(color=color, prefix='fa',
#                            icon='building', marker_anchor=(0, 0))
#   return icon_create

# # Add markers with building icons to the map
# for lat, lng in locations:
#   marker = folium.Marker(location=[lat, lng], icon=create_building_icon())
#   marker.add_to(map)

# Display the map
# map


# streamlit run app.py