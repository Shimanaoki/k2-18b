import os
from arcgis.gis import GIS
import googlemaps
from arcgis.geocoding import geocode
from time import time
import pandas as pd

#環境変数からログイン情報の読み出し
user = os.environ.get("ARCGIS_USER")
password = os.environ.get("ARCGIS_PASS")
url = os.environ.get("ARCGIS_URL")

gis = GIS(url, user, password)

#Google API_Key認証
googleapikey = os.environ.get("GOOGLE_KEY")

gmaps = googlemaps.Client(key=googleapikey)


class my_geocoding():
    
    def __init__(self, csvpath, encoding = None):
        self.table = pd.read_csv(csvpath, encoding)
        try:
            self.N = len(self.table[self.table['score'] == 0])
        except KeyError as e:
            self.N = len(self.table)
            self.table['score'] = 0
            self.table['precision'] = 'BAD'

    def geocode(self, output, address_list, use_googlemaps = True):
    
        self.N = len(self.table[self.table['score'] == 0])
        count = 0
        start = time()
    
        for index, row in self.table[self.table['score'] == 0].iterrows():
    
            count += 1
    
            print('\r','o' * int(index/len(self.table) * 20) + '*' * (20 - int(index/len(self.table) * 20)), round((index/len(self.table))*100,3),'%',end = ' ')
    
            if count % 1000 == 0:
                now = time()
                elasped_time = now - start
                remaining_time = elasped_time/1000 * (self.N - count) / 60
                print(' ', remaining_time, end = ' [m]')
                self.table.to_csv(output, index=False)
                start = time()
    
            try:
                results = geocode(row[address_list[0]])
                self.table.at[index, 'x']= results[0]['location']['x']
                self.table.at[index, 'y'] = results[0]['location']['y']
                self.table.at[index, 'score'] = results[0]['score']
        
            except Exception:
                pass

            if self.table.at[index, 'score'] < 100 and use_googlemaps:

                try:
                    results1 = gmaps.geocode(row[address_list[0]])
                    self.table.at[index, 'x']= results1[0]["geometry"]["location"]["lng"]
                    self.table.at[index, 'y'] = results1[0]["geometry"]["location"]["lat"]
                    self.table.at[index, 'precision'] = results1[0]["geometry"]['location_type']

                except Exception:
                    pass

                if self.table.at[index, 'precision'] != 'ROOFTOP' and len(address_list) > 1:
                    try:
                        results2 = gmaps.geocode(row[address_list[1]])
                        self.table.at[index, 'x']= results2[0]["geometry"]["location"]["lng"]
                        self.table.at[index, 'y'] = results2[0]["geometry"]["location"]["lat"]
                        self.table.at[index, 'precision'] = results2[0]["geometry"]['location_type']
                    
                        if self.table.at[index, 'precision'] != 'ROOFTOP':
                            self.table.at[index, 'x'] = None
                            self.table.at[index, 'y'] = None
                            self.table.at[index, 'precision'] = 'BAD'
                        
                    except Exception:
                        self.table.at[index, 'x'] = None
                        self.table.at[index, 'y'] = None
                        self.table.at[index, 'precision'] = 'BAD'
                        pass
            else:
                self.table.at[index, 'precision'] = 'FINE'
        
        self.table.to_csv(output, index=False)