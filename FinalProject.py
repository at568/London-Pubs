import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

def read_data():
    df = pd.read_csv("PubsDataCS230Final.csv")
    return df

def clean_longitude_latitude(df):
    # [PY1]
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    return df.dropna(subset=['Longitude', 'Latitude'])
#Source https://stackoverflow.com/questions/57286501/why-pd-to-numeric-errors-is-equivalent-to-errors-coerce

def find_closest_location(user_longitude, user_latitude, df):
    closest_location = None
    min_distance = float('inf')
#[DA8]
    for index, row in df.iterrows():
        try:
            location_longitude = float(row['Longitude'])
            location_latitude = float(row['Latitude'])
            distance = measure(user_latitude, user_longitude, location_latitude, location_longitude)

            if distance < min_distance:
                min_distance = distance
                closest_location = row
#[DA7]
        except ValueError:
            st.warning(f"Issue with row {index}: Skipping due to invalid longitude or latitude.")
            continue
#Source for value error https://www.turing.com/kb/valueerror-in-python-and-how-to-fix

    if closest_location is not None:
        closest_location['Distance'] = min_distance
    return closest_location

def measure(lat1, lon1, lat2, lon2):
    R = 6378.137  # Radius of Earth in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(
        dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d
#Source https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula

def plot_density_scatter(df):
    fig = px.density_heatmap(df, x='Longitude', y='Latitude', title='Density Scatter Plot of Pub Locations')
    st.plotly_chart(fig)
# Source https://plotly.com/python/plotly-express/
def plot_scatter_locations(df, user_longitude=None, user_latitude=None, selected_authorities=None):
    #[VIZ1]
    filtered_df = df.copy()

    if selected_authorities:
        filtered_df = df[df['Local Authority'].isin(selected_authorities)]

    plt.figure(figsize=(10, 8))
    plt.scatter(filtered_df['Longitude'], filtered_df['Latitude'], c='blue', edgecolors='w', label='Existing Locations')

    if user_longitude is not None and user_latitude is not None:
        plt.scatter(user_longitude, user_latitude, c='orange', edgecolors='k', s=100, label='Your Location')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Scatter Plot of Pub Locations')
    plt.grid(True)
    plt.legend()

    st.pyplot(plt)
#Source https://plotly.com/python/plotly-express/
def barChart(df, selected_authorities=None):
    #[VIZ2]
    filtered_df = df.copy()
#[PY4] [DA5]
    if selected_authorities:
        filtered_df = df[df['Local Authority'].isin(selected_authorities)]

    fig = px.histogram(filtered_df, x='Local Authority', title='Pubs in each city')
    st.plotly_chart(fig)
# Source https://plotly.com/python/plotly-express/
def pieChart(df, selected_authorities=None):
    #[VIZ3]
    filtered_df = df.copy()
#[PY4] [DA4]
    if selected_authorities:
        filtered_df = df[df['Local Authority'].isin(selected_authorities)]

    fig = px.pie(filtered_df, values=filtered_df.groupby('Local Authority').size(),
                 names=filtered_df['Local Authority'].unique(), title="Pubs in each city")
    #[PY5]
    st.plotly_chart(fig)
#Source https://plotly.com/python/plotly-express/
def plot_box_plot(df, selected_authorities=None):
    filtered_df = df.copy()

    if selected_authorities:
        filtered_df = df[df['Local Authority'].isin(selected_authorities)]

    fig = px.box(filtered_df, x="Local Authority", y="Profitable", title="Profitability by Local Authority")
    st.plotly_chart(fig)
#Source https://plotly.com/python/plotly-express/
def get_pub_information(pub_name, df):
    #[DA9]
    pub_name_lower = pub_name.lower()
    df['Name_lower'] = df['Name'].str.lower()

    pub_info = df[df['Name_lower'] == pub_name_lower]
    if len(pub_info) > 0:
#[PY2]
        return pub_info.iloc[0]['Longitude'], pub_info.iloc[0]['Latitude'], pub_info.iloc[0]['Profitable']
    else:
        return None, None, None
# Source https://www.geeksforgeeks.org/python-extracting-rows-using-pandas-iloc/
def get_most_profitable_pub(df):
    most_profitable = df.loc[df['Profitable'].idxmax()]
    return most_profitable['Name'], most_profitable['Profitable']

def main():
    #[ST4]
    st.title("London Pubs")

    #[ST4]
    sidebar_option = st.sidebar.selectbox("Navigation", ["Home Page", "Find Nearest Pub", "Search Pub", "Visualizations", "Add a Pub"])

    profitable_only = st.sidebar.checkbox("Show Only Profitable Firms")
    df = read_data()
    df_cleaned = clean_longitude_latitude(df)
    #[PY3], [DA1]
    if profitable_only:
        df_cleaned = df_cleaned[df_cleaned['Profitable'] > 0]

    if sidebar_option == "Home Page":
        st.header("Home Page - London Pubs")
        st.image("PubsImage.jpg", caption='Star Tavern in London')
        #[DA3]
        most_profitable_pub, max_profit = get_most_profitable_pub(df_cleaned)
        st.write(f"Congratulations to {most_profitable_pub} for being the most profitable pub of the year!")
        st.write(f"Profitability: £{max_profit}")
        st.text("Look at all the firms in the Dataset")
        st.write(df_cleaned)

    elif sidebar_option == "Find Nearest Pub":
        st.header("Find the Nearest Pub")
        df = read_data()
        df_cleaned = clean_longitude_latitude(df_cleaned)
        user_longitude = st.sidebar.number_input("Enter your longitude:")
        user_latitude = st.sidebar.number_input("Enter your latitude:")
        selected_authorities_scatter = st.sidebar.multiselect("Select Local Authorities", df_cleaned['Local Authority'].unique())
        plot_scatter_locations(df_cleaned, user_longitude, user_latitude, selected_authorities_scatter)
#[PY3]
    elif sidebar_option == "Search Pub":
        st.header("Search for a Pub")
        pub_name = st.sidebar.text_input("Enter the name of the pub:")
        if st.sidebar.button("Search"):
            longitude, latitude, profitability = get_pub_information(pub_name, df_cleaned)
            if longitude is not None:
                st.success(f"Found pub '{pub_name}'!")
                st.write(f"Longitude: {longitude}")
                st.write(f"Latitude: {latitude}")
                st.write(f"Profitability: {profitability}")
            else:
                st.error(f"Pub '{pub_name}' not found.")

    elif sidebar_option == "Visualizations":
        st.header("Visualizations")
        df = read_data()
        df_cleaned = clean_longitude_latitude(df_cleaned)
        viz_option = st.sidebar.selectbox("Select a Visualization", ["Density Scatter Plot", "Bar Chart", "Pie Chart"])
        if viz_option == "Density Scatter Plot":
            plot_density_scatter(df_cleaned)
        elif viz_option == "Bar Chart":
            selected_authorities_bar = st.sidebar.multiselect("Select Local Authorities", df_cleaned['Local Authority'].unique())
            barChart(df_cleaned, selected_authorities_bar)
        elif viz_option == "Pie Chart":
            selected_authorities_pie = st.sidebar.multiselect("Select Local Authorities", df_cleaned['Local Authority'].unique())
            pieChart(df_cleaned, selected_authorities_pie)

    elif sidebar_option == "Add a Pub":
        st.header("Add a New Pub")
        new_pub_name = st.text_input("Enter the name of the new pub:")
        new_pub_authority = st.text_input("Enter the local authority of the new pub:")
        new_pub_longitude = st.number_input("Enter the longitude of the new pub:")
        new_pub_latitude = st.number_input("Enter the latitude of the new pub:")
        new_pub_profitability = st.number_input("Enter the profitability of the pub:", value=0)

        if st.button("Add Pub"):
            df = read_data()
            new_row = {
                'Name': new_pub_name,
                'Local Authority': new_pub_authority,
                'Longitude': new_pub_longitude,
                'Latitude': new_pub_latitude,
                'Profitability': new_pub_profitability,
            }
            df = df.append(new_row, ignore_index=True)
            st.success(f"New pub '{new_pub_name}' added successfully!")
            st.write(df)  # Display updated dataframe

if __name__ == '__main__':
    main()
