# import libraries
import folium
import pandas as pd

# Make a data frame with dots to show on the map
# import libraries
import folium
import pandas as pd

# Make a data frame with dots to show on the map
data = pd.DataFrame({
    'lat': [76.81, 2],

'name': ['Paris', 'melbourne' ],
    'lon': [9.38, 49, ]

})
data

# Make an empty map
m = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)

# I can add marker one by one on the map
for i in range(0, len(data)):
    folium.Marker([data.iloc[i]['lon'], data.iloc[i]['lat']], popup=data.iloc[i]['name']).add_to(m)

# Save it as html
m.save('312_markers_on_folium_map1.html')

print(data)
# Make an empty map
m = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)

# I can add marker one by one on the map
for i in range(0, len(data)):
    folium.Marker([data.iloc[i]['lon'], data.iloc[i]['lat']], popup=data.iloc[i]['name']).add_to(m)

# Save it as html
m.save('312_markers_on_folium_map1.html')
