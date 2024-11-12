### crear envirpnment: python -m venv venv
## activate: venv\Scripts\activate

import numpy as np
import pandas as pd

### Lectura de Datos - Data reading
cities = pd.read_csv("Data/cities.csv")
constellations=pd.read_csv("Data/constellations.csv")
meteorshowers=pd.read_csv("Data/meteorshowers.csv")
moonphases=pd.read_csv("Data/moonphases.csv")


### Add Chang'e 
change_meteor_shower = {'name':'Chang\'e','radiant':'Draco','bestmonth':'october','startmonth':'october','startday':1,'endmonth':'october','endday':31,'hemisphere':'northern','preferredhemisphere':'northern'}

draco_constellation = {'constellation':'Draco','bestmonth':'july','latitudestart':90,'latitudeend':-15,'besttime':2100,'hemisphere':'northern'}

change_meteor_shower = {'name':'Chang\'e','radiant':'Draco','bestmonth':'october','startmonth':'october','startday':1,'endmonth':'october','endday':31,'hemisphere':'northern','preferredhemisphere':'northern'}

meteorshowers = pd.concat([meteorshowers, pd.DataFrame(change_meteor_shower, index=[0])], ignore_index=True)


draco_constellation = {'constellation':'Draco','bestmonth':'july','latitudestart':90,'latitudeend':-15,'besttime':2100,'hemisphere':'northern'}

constellations = pd.concat([constellations, pd.DataFrame(draco_constellation, index=[0])], ignore_index=True)


### Data Exploration:
## eplore each dataset:


cities.head()


cities.info()
constellations.head()
constellations.info()
meteorshowers.head()
meteorshowers.info()
moonphases.head()
moonphases.info()

# constellations["bestmonth"].unique()

## Meses del año y codigo numérico   # Month of the year with their code.
meses = {'january': 1, 'february':2, 'march':3, 'april':4, 'may':5, 'june':6 , 'july': 7, 'august':8, 'september':9,
         'october':10, 'november':11, 'december':12}

### Change the names of the months by the number:
meteorshowers.bestmonth = meteorshowers.bestmonth.map(meses)
meteorshowers.startmonth = meteorshowers.startmonth.map(meses)
meteorshowers.endmonth = meteorshowers.endmonth.map(meses)
moonphases.month = moonphases.month.map(meses)
constellations.bestmonth = constellations.bestmonth.map(meses)

#Verificar los cambios:
meteorshowers["bestmonth"].unique()
meteorshowers["startmonth"].unique()
meteorshowers["endmonth"].unique()
moonphases["month"].unique()
constellations["bestmonth"].unique()

### Convert to date the columns: startdate and enddate and add them to the dataframe: meteorshowers & moonphases
meteorshowers.info()

meteorshowers['startdate']= pd.to_datetime( (2020 * 10000) + (meteorshowers.startmonth * 100) + (meteorshowers.startday ) , format= '%Y%m%d' )
meteorshowers['startdate'].unique()

meteorshowers['enddate'] = pd.to_datetime(2020*10000+meteorshowers.endmonth*100+meteorshowers.endday,format='%Y%m%d')
meteorshowers['enddate'].unique()

moonphases.info()

moonphases['date'] = pd.to_datetime(2020*10000+moonphases.month*100+moonphases.day,format='%Y%m%d')

## Convert the hemispher in numbers:
hemispheres = {'northern':0, 'southern':1, 'northern, southern':3}
meteorshowers.hemisphere = meteorshowers.hemisphere.map(hemispheres)
constellations.hemisphere = constellations.hemisphere.map(hemispheres)

#constellations['hemisphere'].unique

### Convert the phases of the moon in numbers:

phases_of_the_moon= {'new moon':0,'third quarter':0.5, 'first quarter':0.5,'full moon':1.0}
moonphases['percentage'] = moonphases.moonphase.map(phases_of_the_moon) 
moonphases.head()


## Delete in unnecessary data:
###  delete the following columns from the datasets:
### meteorshowers:	startmonth, startday, endmonth, endday, hemisphere
### moonphases	month, day, moonphase, specialevent
###constellations	besttime

meteorshowers.info()
moonphases.info()

meteorshowers = meteorshowers.drop(['startmonth', 'startday', 'endmonth', 'endday', 'hemisphere'], axis=1)
moonphases = moonphases.drop(['month','day','moonphase','specialevent'], axis=1)
constellations = constellations.drop(['besttime'], axis=1)


### Guardas la ultim fase. #Save the las value of phase:

UltimaFase=0
#moonphases

for index, row in moonphases.iterrows():
    if pd.isnull(row['percentage']):
        moonphases.at[index,'percentage'] = UltimaFase
    else:
        UltimaFase = row['percentage']

moonphases.info()
moonphases.head()


#### Forecasting function.-
### we need to determine the latitud. 
### Use the latitud to determine which constellations are seen in the cities
### use the constellations to see metero shower than can be seen in the cities
### use the meteor shower to determine the dates on the constellations are visible.
## use the dates to search the optimal date in which you perceive less quantity of the moon's light
 
meteorshowers.info()
moonphases.info()
cities.info()
constellations.info()

#print(cities['city'] == 'Abu Dhabi')
#print(cities['city'] == 'Abuja')

def predict_best_meteor_shower_viewing(city):
    # Create an empty string to return the message back to the user
    meteor_shower_string = ""
    
    if city not in cities.values:
        meteor_shower_string = "Unfortunately, " + city + " isn't available for a prediction at this time."
        return meteor_shower_string

    # Get the latitude of the city from the cities DataFrame
    latitude = cities.loc[cities['city'] == city, 'latitude'].iloc[0]
    
    # List of constellations that are viewable from that latitude
    constellation_list = constellations.loc[(constellations['latitudestart'] >= latitude) & (constellations['latitudeend'] <= latitude), 'constellation'].tolist()
    
    # If no constellations are viewable, let the user know
    if not constellation_list:
        meteor_shower_string = "Unfortunately, there are no meteor showers viewable from "+ city + "."

        return meteor_shower_string

    meteor_shower_string = "In " + city + " you can see the following meteor showers:\n"
    
    
    # Iterate through each constellation in constellation list
    for constellation in constellation_list:
        # Find the meteor shower that is nearest to that constellation
        meteorshower = meteorshowers.loc[meteorshowers['radiant'] == constellation, 'name'].iloc[0]
        # Find the start and end dates for that meteor shower
        meteorShowerStartdate = meteorshowers.loc[meteorshowers['radiant'] == constellation, 'startdate'].iloc[0]
        meteorShowerEnddate = meteorshowers.loc[meteorshowers['radiant'] == constellation, 'enddate'].iloc[0]

        # Find the Moon phases for each date within the viewable time frame of that meteor shower
        moon_phases_list = moonphases.loc[(moonphases['date'] >= meteorShowerStartdate) & (moonphases['date'] <= meteorShowerEnddate)]
        
        if meteorshower == 'Chang\'e':
            # For the film meteor shower, find the date where the Moon is the most visible
            best_moon_date = moon_phases_list.loc[moon_phases_list['percentage'].idxmax()]['date']

            # Add that date to the string to report back to the user
            meteor_shower_string += "Though the Moon will be bright, " + meteorshower + "'s meteor shower is best seen if you look towards the " + constellation + " constellation on " +  best_moon_date.to_pydatetime().strftime("%B %d, %Y") + ".\n"
        else:
        # Find the first date where the Moon is the least visible
            best_moon_date = moon_phases_list.loc[moon_phases_list['percentage'].idxmin()]['date']

            # Add that date to the string to report back to the user
            meteor_shower_string += meteorshower + " is best seen if you look towards the " + constellation + " constellation on " +  best_moon_date.to_pydatetime().strftime("%B %d, %Y") + ".\n"
    
    return meteor_shower_string



print(predict_best_meteor_shower_viewing('Beijing'))


