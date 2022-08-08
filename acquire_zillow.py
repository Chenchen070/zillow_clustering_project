import pandas as pd
import env

def get_connection(db, username=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'
    
sql_query = '''
SELECT basementsqft as basement_square_ft, bathroomcnt as bathroom, bedroomcnt as bedroom, parcelid,
buildingqualitytypeid as quality_type, calculatedfinishedsquarefeet as finished_square_ft, garagecarcnt as garage,
fips, latitude, longitude, lotsizesquarefeet as lot_square_ft, regionidcity as city, poolcnt as pool,
yearbuilt, structuretaxvaluedollarcnt as structure_value, taxvaluedollarcnt as house_value,
landtaxvaluedollarcnt as land_value, taxamount as tax, logerror as log_error, transactiondate as transaction_date
FROM properties_2017
JOIN predictions_2017 USING (parcelid)
WHERE transactiondate < '2018'
AND propertylandusetypeid = 261
'''

def get_zillow_data():
    df = pd.read_sql(sql_query, get_connection('zillow'))
    return df