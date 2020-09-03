import pandas as pd
import numpy as np

#Import data
df = pd.read_hdf(r'..\..\data\clean_data.hdf',key='df')

df.sort_values(['consumer_id','year_y'])
df['choice_set'] = df.groupby(['insurer_id', 'year_y'])['provider_id'].transform('unique') #Create choice set
df['Y'] = 0 #Add the Y variable
df = df.explode('choice_set') #Expand the rows according to the choice set
df = df.rename(columns = {"choice_set":'choice'})
df.loc[df['choice']==df['provider_id'],'Y'] = 1 #Chosen alternative has Y=1

#Unit test for assignment
assert sum(df['Y']) == len(pd.read_hdf(r'..\..\data\clean_data.hdf',key='df'))
print("Unit test 1 passed: Correct number of choices.")


# Not needed anymore - different method used
# df['tile'] = df['choice_set'].str.len()
# #Tile the data to fit the choice set
# df = df.loc[df.index.repeat(df.tile)] #Repeat each row to match the length of the choice set
# #Make a unit test to check choice_sets, test for 100 random values
# for i in np.random.choice(len(df),size=100):
#     assert (df.iloc[[i]]['choice_set'].values[0] == df.groupby(['insurer_id', 'year_y'])['provider_id'].unique().loc[(df.iloc[[i]]['insurer_id'], df.iloc[[i]]['year_y'])].values[0]).all()
# print("Unit test 1 passed: Choice set not found to be incorrect.")


#Make a provider df with relevant data
provider_df = df[['price', 'year_y', 'provider_id','public','insurer_id']].copy()
provider_df = provider_df.rename(columns={'provider_id':'choice'})
provider_df = provider_df.drop_duplicates()

#Merge the two df's on year and choice taking the values for price and public from provider_df
df = df.merge(provider_df, left_on=['choice','year_y','insurer_id'], right_on=['choice','year_y','insurer_id'], how='left')

#Unit test if prices are correct for chosen alternatives
assert sum(df.loc[df['Y']==1, 'price_x']-df.loc[df['Y']==1, 'price_y'])==0
print("Unit test 2 passed: Prices were correctly drawn from provider_df.")


df = df.drop(columns=['year_x', 'index_x', 'price_negotiated','year_negotiated', 'index_y',
                      'price_provider','public_x','price_x']) #Drop unnecesary columns used in cleaning
df = df.rename(columns={"year_y":"year", 'price_y':"price",'public_y':'public'})

#Make the older than median in the previos year in a given hospital dummy variable
df['median_age'] = df.groupby(['provider_id','year'])['age'].transform('median')
provider_df = df.drop_duplicates(subset=['year','provider_id'])[['provider_id','year','median_age']]
provider_df = provider_df.sort_values(['provider_id','year'])
provider_df['median_age_shifted'] = provider_df.groupby('provider_id')['median_age'].shift(1)
provider_df = provider_df.drop(columns=['median_age']) #Drop unnecesary columns used in cleaning

#Add the median_age_shifted column to the original df
df = df.merge(provider_df, left_on=['provider_id','year'], right_on=['provider_id','year'], how='left')
df = df.loc[df['year']>2013] #Drop all values for the first year
df['older'] = 0
df.loc[df['age']>=df['median_age_shifted'],'older'] = 1
df = df.drop(columns=['median_age','median_age_shifted']) #Drop unnecesary columns used in cleaning
df = df.sort_values(['consumer_id','year'])

#Export the data
df.to_hdf(r'..\..\data\clean_data.hdf', key='data',mode='a')