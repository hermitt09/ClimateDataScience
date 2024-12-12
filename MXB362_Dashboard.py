## Import Necessary Libraries
# File and Data Handling
import os
import io
from netCDF4 import Dataset
import netCDF4 as nc
import h5py
from datetime import datetime
import time as time_lib

# Data Manipulation and Analysis
import numpy as np
import xarray as xr
import pandas as pd
import dask.array as da

# Geospatial Mapping
from folium.plugins import HeatMap, HeatMapWithTime
import rasterio
from rasterio.transform import from_origin

# Data Visualisation
import matplotlib.pyplot as plt
import matplotlib.pyplot as mplot
import matplotlib
import matplotlib.image as mpimg
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from PIL import Image
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go
import imageio

# Colour Mapping and Styling
import matplotlib.colors as mcolors
import branca.colormap as cm

# Dashboard Function
import streamlit as st
import folium
from streamlit_folium import folium_static


# Define file paths
nc_fpaths = [
    r"Peatland_carbon_balance_1382_1-20240926_104125\fluc_kk10.nc", 
    r"Peatland_carbon_balance_1382_1-20240926_104125\fluc_hyde31u.nc", 
    r"Peatland_carbon_balance_1382_1-20240926_104125\fluc_hyde32u.nc",  
    r"Peatland_carbon_balance_1382_1-20240926_104125\fluc_hyde31.nc", 
    r"Peatland_carbon_balance_1382_1-20240926_104125\fluc_hyde32.nc", 
    r"Peatland_carbon_balance_1382_1-20240926_104125\fluc_kk10d.nc"
]
base_path = r"rss1windnv7r01_7R01-20240914_111905"
wind_speed_file = os.path.join(base_path, "ws_v07r01_202212.nc4.nc")
ice_thickness = Dataset(r"C:\Users\gksho\OneDrive - Queensland University of Technology\2024-S~2\MXB362\ASSIGN~1\MXB362~1\MXB362~4\NSIDC-~1\NSIDC-~1.NC", 'r')

h5file_early = "LEOLSTCMG30_002-20240919_092049\LEOLSTCMG30_200212_002_20230823132537.h5"  
h5file_late = "LEOLSTCMG30_002-20240919_092049\LEOLSTCMG30_202212_002_20230823134940.h5" 
h5file = "LEOLSTCMG30_002-20240919_092049/LEOLSTCMG30_200208_002_20230823132515.h5"


############################################################################################################

def plot_land_temp_comparison():
    # Sidebar with information
    st.sidebar.header("Why Small Changes in Temperature Matter")
    st.sidebar.write(""" The two plots here compare the land surface temperature 20 years apart. While at first glance there appears to be minimal changes, when looking at any given section we can see that there is an small increase in every section. You may not think this is bad, however, even the smallest changes in temperature can have significant effects on the climate. 

A rise of just 1°C can lead to more extreme weather events, altered ecosystems, and rising sea levels.  For instance, the Intergovernmental Panel on Climate Change (IPCC) reports that global temperature increases lead to severe impacts on biodiversity and the frequency of heatwaves.  Studies have shown that ecosystems are highly sensitive to minute temperature changes, which affects species distribution and food webs. 

**References:**
- Intergovernmental Panel on Climate Change (IPCC). "Climate Change 2021: The Physical Science Basis."
- Parmesan, C., & Yohe, G. (2003). "A globally coherent fingerprint of climate change impacts across natural systems."
""")
    # Function to load and normalize LST data
    def load_and_normalize(h5file):
        with h5py.File(h5file, 'r') as file:
            LST = file['/LST_Day_QDG'][:]
        # Normalize data to the range [0, 1] for colormap
        LST_normalized = (LST - np.min(LST)) / (np.max(LST) - np.min(LST))
        return LST_normalized
    
    # Load data for the earliest and latest files
    LST_early = load_and_normalize(h5file_early)
    LST_late = load_and_normalize(h5file_late)

    # Apply colormap
    colormap = plt.get_cmap('viridis')
    colored_image_early = colormap(LST_early)[:, :, :3]  # Drop alpha and keep RGB
    colored_image_late = colormap(LST_late)[:, :, :3]

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    # Plot for the earliest time period
    axs[0].imshow(colored_image_early)
    axs[0].set_title('Land Surface Temperature - December 2002')
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')

    # Plot for the latest time period
    axs[1].imshow(colored_image_late)
    axs[1].set_title('Land Surface Temperature - December 2022')
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Latitude')
    # Adjust layout and add colorbars
    #plt.tight_layout()
    plt.colorbar(axs[0].imshow(colored_image_early), ax=axs[0], label='Temperature Value', shrink = 0.5)
    plt.colorbar(axs[1].imshow(colored_image_late), ax=axs[1], label='Temperature Value', shrink = 0.5)
    plt.subplots_adjust(wspace=0.3)  # Adjust the spacing between plots
    #fig.update_layout(width = 800, height = 600)
    # Display the plot
    st.pyplot(fig, use_container_width = True)
    
# Create a Folium map with the Esri World Imagery base
def create_map():
    m = folium.Map(location=[0, 0], 
                   tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", 
                   attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
                   zoom_start=2)
    return m

##########################################################################################################################
# Plot CO2 emissions using HeatMapWithTime on Folium map
def plot_co2_emissions_on_map(folium_map, time_dim):
    
    st.sidebar.subheader("CO2 Emissions Throughout The Holocene")
    st.sidebar.write("""The Holocene CO2 plot covers approximately the last 10,000 years, showing the natural fluctuations of CO2 concentrations in a relatively stable climate period. Throughout most of the Holocene, CO2 levels remained within a narrow range as human civilisations evolved, agriculture developed, and ecosystems adapted to the conditions. However, the latter part of this period, especially after 1750, shows a dramatic departure from the stable historical levels, coinciding with the onset of the Industrial Revolution. 
    
The data is measured in gC m-2 (10 yer)-1. It shows the average carbon (gC) in a given area (m-2) over a period of 10 years (10 yer)-1.

This plot starkly contrasts the slow natural fluctuations of CO2 with the rapid, unprecedented spike seen in recent centuries. This deviation underscores the role of human activity in disrupting the Earth's natural carbon cycle and is a major indicator of the anthropogenic drivers behind the current climate crisis. While the data does not show the last 30 years, the second figure shows the continued steep rise in emissions""")
    
    # Prepare the data
    fLUC = dat1.variables['fLUC']
    lats = dat1.variables['LATITUDE'].values
    lons = dat1.variables['LONGITUDE'].values
    time_dim = dat1.dims['TIME']

    # Adjust longitudes to be between -180 and 180
    lons = np.where(lons > 180, lons - 360, lons)
    
    heat_data_per_time = []
    
    # Loop through each time step and prepare the heatmap data
    for time_index in range(fLUC.shape[1]):
        current_data = fLUC.isel(TIME=time_index).mean(dim='scenario').compute().values
        heat_data = [[lats[lat_idx], lons[lon_idx], current_data[lat_idx, lon_idx]]
                     for lat_idx in range(len(lats))
                     for lon_idx in range(len(lons))
                     if current_data[lat_idx, lon_idx] > 0]  # Only include non-zero data points
        
        heat_data_per_time.append(heat_data)
    
        # Create a gradient dictionary for the HeatMap
    gradient = {
        0.0: '#0d0887',  # Low values (dark purple)
        0.2: '#6a00a8',
        0.4: '#b12a90',
        0.6: '#e16462',
        0.8: '#fca636',
        1.0: '#f0f921'  # High values (yellow)
    }

    adjusted_time_periods = [-10000 + 10 * i for i in range(time_dim)]
    
    # Add heatmap with time
    HeatMapWithTime(heat_data_per_time, index=adjusted_time_periods,
                    auto_play=True, 
                    max_opacity=0.7, 
                    gradient=gradient, 
                    use_local_extrema=True).add_to(folium_map)

    # Create the colour map for the colour bar.
    colormap = cm.LinearColormap(['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921'], vmin=0, vmax=1)
    colormap.caption = 'CO2 Emissions (gC m-2 (10 yr)-1)'

    # Add the colormap to the Folium map
    folium_map.add_child(colormap)
    
    folium_static(folium_map)

    
    

######################################################################################################
def plot_co2_line_graph():
    #Load the data
    co2_data = pd.read_csv('CO2data.csv')

    # Create a DataFrame
    df = pd.DataFrame(co2_data)

    # Create the line plot with Plotly
    fig_co2 = go.Figure()

    # Add monthly average line
    fig_co2.add_trace(go.Scatter(x=df['Year'], y=df['monthly average'], mode='lines', name='Monthly Average'))

    # layout
    fig_co2.update_layout(
        title='CO2 Global Emissions (1958 - 2022)',
        xaxis_title='Year',
        yaxis_title='CO2 (ppm)',
        showlegend=True
    )
    st.sidebar.subheader("CO2 Emissions Throughout Recent History (1958 - 2022)")
    st.sidebar.write("""This plot displays the monthly average CO2 concentrations in the atmosphere from 1958 to 2022, providing a critical perspective on recent anthropogenic climate change. The sharp rise in CO2 levels since the mid-20th century marks the beginning of the industrial era, driven by the widespread use of fossil fuels, deforestation, and other human activities. Carbon dioxide is a potent greenhouse gas, and its rapid increase correlates strongly with global warming and the intensification of extreme weather patterns.
                     
As the plot shows, there is a steady upward trend with seasonal fluctuations due to natural cycles like plant respiration and ocean uptake. However, the overall rise in atmospheric CO2 represents a primary driver of climate change, leading to more heat being trapped in Earth's atmosphere, contributing to rising global temperatures.""")
    # Show the interactive plot
    st.plotly_chart(fig_co2)

##############################################################################################################
# Plot wind speed on a Folium map
def plot_wind_speed_on_map():
    st.sidebar.subheader("Wind Speed Comparison")
    st.sidebar.write("""This visualisation is crucial for understanding the impacts of climate change on wind patterns. By comparing wind speed data from 1988 and 2022, we can observe changes that may correlate with the rising global temperatures and altering weather patterns due to increased greenhouse gas emissions. Such visual evidence helps reinforce the urgency of addressing climate change. For example, the Intergovernmental Panel on Climate Change (IPCC) reports how human activities significantly influence atmospheric patterns (IPCC, 2021). By visually representing these changes, we can better communicate the science behind climate change to policymakers and the public, pushing for informed and effective action.

*References:*

IPCC. (2021). Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change. Cambridge University Press""")
    
    def read_wind_speed_data(file_path):
        with Dataset(file_path) as d:
            lats = d.variables['latitude'][:]
            lons = d.variables['longitude'][:]
            wind_speed = d.variables['wind_speed'][:].astype(np.float32)
            wind_speed[wind_speed == -999] = np.nan  # Handle missing values
            lons = np.where(lons > 180, lons - 360, lons)  # Adjust longitudes
        return wind_speed, lats, lons

    def plot_wind_speed_comparison(file1, file2, year1, year2):
        wind_speed1, lats, lons = read_wind_speed_data(file1)
        wind_speed2, _, _ = read_wind_speed_data(file2)
    
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
        # Define color scale limits
        vmin = min(np.nanmin(wind_speed1), np.nanmin(wind_speed2))
        vmax = max(np.nanmax(wind_speed1), np.nanmax(wind_speed2))
    
        mp1 = Basemap(projection='robin', lon_0=0, llcrnrlat=-90, urcrnrlat=90,
                      llcrnrlon=-180, urcrnrlon=180, resolution='i', ax=axes[0])
        mp2 = Basemap(projection='robin', lon_0=0, llcrnrlat=-90, urcrnrlat=90,
                      llcrnrlon=-180, urcrnrlon=180, resolution='i', ax=axes[1])
    
        lon, lat = np.meshgrid(lons, lats)
        x, y = mp1(lon, lat)
    
        c_scheme1 = mp1.pcolormesh(x, y, np.squeeze(wind_speed1), cmap='Purples', shading='auto', vmin=vmin, vmax=vmax)
        mp1.drawcoastlines()
        mp1.drawcountries()
        mp1.drawstates()
        mp1.shadedrelief()
        mp1.etopo()
        axes[0].set_title(f"Wind Speed in {year1}")
    
        x, y = mp2(lon, lat)
        c_scheme2 = mp2.pcolormesh(x, y, np.squeeze(wind_speed2), cmap='Purples', shading='auto', vmin=vmin, vmax=vmax)
        mp2.drawcoastlines()
        mp2.drawcountries()
        mp2.drawstates()
        mp2.shadedrelief()
        mp2.etopo()
        axes[1].set_title(f"Wind Speed in {year2}")
    
        # Create a single colorbar
        cbar = fig.colorbar(c_scheme1, ax=axes, orientation='horizontal', fraction=0.03, pad=0.05)
        cbar.set_label('Wind Speed (m/s)')
    
        st.pyplot(fig)

    file1 = r"rss1windnv7r01_7R01-20240914_111905\ws_v07r01_198812.nc4.nc"
    file2 = r"rss1windnv7r01_7R01-20240914_111905\ws_v07r01_202212.nc4.nc"
    
    st.title("Wind Speed Comparison: 1988 vs. 2022")
    plot_wind_speed_comparison(file1, file2, 1988, 2022)
        
##########################################################################################################################
def plot_land_temp_animation():
       # Sidebar with information
    st.sidebar.header("Why Small Changes in Temperature Matter")
    st.sidebar.write(""" The animation here compares the land surface temperature over the last 20 years. While at first glance there appears to be minimal changes, when looking at any given section we can see that there is small changes on every continent. You may not think this is bad, however, even the smallest changes in temperature can have significant effects on the climate. 

A rise of just 1°C can lead to more extreme weather events, altered ecosystems, and rising sea levels.  For instance, the Intergovernmental Panel on Climate Change (IPCC) reports that global temperature increases lead to severe impacts on biodiversity and the frequency of heatwaves.  Studies have shown that ecosystems are highly sensitive to minute temperature changes, which affects species distribution and food webs. 

**References:**
- Intergovernmental Panel on Climate Change (IPCC). "Climate Change 2021: The Physical Science Basis."
- Parmesan, C., & Yohe, G. (2003). "A globally coherent fingerprint of climate change impacts across natural systems."
""")
    # Path to the folder containing your HDF5 files
    folder_path = "LEOLSTCMG30_002-20240919_092049"

    # Get sorted list of HDF5 files
    files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5')])

    # Check if we have the correct number of files
    if len(files) != 245:
        st.error(f"Expected 245 HDF5 files but found {len(files)} files.")
        st.stop()

    # Start date
    start_date = datetime(2002, 8, 1)  # Starting from August 2002
    date_labels = [start_date + pd.DateOffset(months=i) for i in range(len(files))]

    def load_lst_data(h5_file):
        # This function loads the data from a single .h5 file
        with h5py.File(h5_file, 'r') as file:
            # Load the dataset from the HDF5 file
            LST = file['/LST_Day_HDG'][:]
            # Normalize the LST data for color scaling
            LST_normalized = (LST - np.nanmin(LST)) / (np.nanmax(LST) - np.nanmin(LST))
        return LST_normalized

    # Create the figure and Basemap
    fig, ax = plt.subplots(figsize=(12, 8))
    m = Basemap(projection='moll', lon_0=0, resolution='c', ax=ax)

    # Create a grid for latitudes and longitudes
    lat = np.linspace(90, -90, 360)  # Flip latitude to correct upside-down view
    lon = np.linspace(-180, 180, 720)  # Adjust longitude from -180 to 180
    lon, lat = np.meshgrid(lon, lat)
    x, y = m(lon, lat)

    # Initialize the colormesh with the first dataset
    LST_normalized = load_lst_data(files[0])
    cmesh = m.pcolormesh(x, y, LST_normalized, shading='auto', cmap='viridis', ax=ax)

    # Add the initial colorbar and title
    colorbar = plt.colorbar(cmesh, ax=ax, label='Temperature (Kelvins)')
    ax.set_title(f"Land Surface Temperature - {date_labels[0].strftime('%b %Y')}")

    # Function to update the plot for each frame
    def update(frame):
        LST_normalized = load_lst_data(files[frame])
        cmesh.set_array(LST_normalized.ravel())
        cmesh.set_clim(vmin=0, vmax=1)  # Reset color limits to ensure proper scaling
        ax.set_title(f"Land Surface Temperature - {date_labels[frame].strftime('%b %Y')}")
        return cmesh, ax

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(files), blit=False, repeat=True)

    # Save the animation as an MP4 file
    writer = FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("land_surface_temperature.mp4", writer=writer)

    # Load and display the video in Streamlit
    st.video("land_surface_temperature.mp4")

############################################################################################################################
def plot_ice_thickness():
    st.sidebar.subheader("Ice Thickness")
    st.sidebar.write("""The polar regions, particularly Antarctica, play a crucial role in regulating Earth's climate and sea levels. The plot you have visualises key indicators of ice dynamics in the Antarctic region, specifically ice thickness and the velocity of ice flow.

**Ice Volume and Sea Level Rise: **

The Antarctic ice sheet contains approximately 60% of the world’s freshwater. Changes in ice thickness directly impact global sea levels. Even small variations in the volume of this ice can result in significant sea-level changes, threatening coastal communities worldwide (Ice Sheets (Antarctic), n.d.).

**Indicators of Climate Change: **

A decrease in ice thickness over time serves as a clear indicator of climate change. As global temperatures rise due to increased greenhouse gas emissions, ice melts at an accelerated pace. Monitoring ice thickness helps scientists understand the rate of this melting and its future implications.

**Understanding Ice Flow Dynamics**

**Velocity Field:** The plot also includes a quiver plot that represents the velocity of the ice flow. This data indicates how quickly ice is moving from the interior of the Antarctic ice sheet to the coast. Accelerated flow rates can signal the destabilisation of the ice sheet, leading to increased melting and calving events (US EPA, 2021).

**Feedback Mechanisms:** As ice melts and the underlying surface becomes exposed, darker ocean water is revealed. This lowers the albedo (reflectivity) of the surface, causing more heat absorption and further accelerating ice loss. Understanding the velocity of ice flow helps identify regions most at risk of entering this feedback loop (US EPA, 2021).


**References:**

Ice Sheets (Antarctic). (n.d.). ESA Climate Office. https://climate.esa.int/en/projects/ice-sheets-antarctic/ 

US EPA. (2021, March 18). Climate Change Indicators: Ice Sheets. Www.epa.gov. https://www.epa.gov/climate-indicators/climate-change-indicators-ice-sheets""")
    # Load data
    x = ice_thickness.variables['x'][:]
    y = ice_thickness.variables['y'][:]
    vx = ice_thickness.variables['vx'][:]
    vy = ice_thickness.variables['vy'][:]
    thickness = ice_thickness.variables['thickness'][:]

    # Plot setup
    ice_fig = plt.figure(figsize=(15, 15))
    plt.title('Velocity Field and Ice Thickness')

    # Ice thickness as the background 
    img = plt.imshow(thickness, cmap='coolwarm', extent=[x.min(), x.max(), y.min(), y.max()])

    # Quiver plot for velocity
    skip = 200  # To reduce the density of arrows
    plt.quiver(x[::skip], y[::skip], vx[::skip, ::skip], vy[::skip, ::skip], color='black', width = 0.0015)

    plt.colorbar(img, label='Ice Thickness (m)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    st.pyplot(ice_fig)

###########################################################################################################################
# Streamlit App Layout
st.title("Interactive Climate Data Visualisation")

# Create a selection box to choose between CO2 emissions or wind speed
option = st.selectbox('Choose a climate indicator to display:', ('CO2 Emissions', 'Wind Speed', 'Land Temperature', 'Antarctic Ice'))

# Generate the base map
folium_map = create_map()

if option == 'CO2 Emissions':
    # Load the dataset to get time dimensions
    # Streamlit App Layout 
    st.title("CO2 Emissions Throughout The Holocene")
    dat1 = xr.open_mfdataset(nc_fpaths, coords='minimal', combine='nested', concat_dim='scenario', engine='netcdf4')
    time_dim = dat1.dims['TIME']

    # Plot the CO2 emissions for the selected time index
    plot_co2_emissions_on_map(folium_map, time_dim)
    
    st.title("CO2 Change from 1958 to 2022")
    plot_co2_line_graph()
    
    st.write('Data Sourced from: Stocker, B.D., Z. Yu, C. Massa, and F. Joos. 2017. Global Peatland Carbon Balance and Land Use Change CO2 Emissions Through the Holocene. ORNL DAAC, Oak Ridge, #Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/1382')
   
elif option == 'Wind Speed':
    # Plot the wind speed data
    plot_wind_speed_on_map()

elif option == 'Land Temperature':
    #plot_land_temp() 
    plot_land_temp_animation()
else:
    plot_ice_thickness()
