import pandas as pd
import numpy as np

#Importing dataframes
df = pd.read_stata(r'..\..\data\choices.dta')
df_prices = pd.read_stata(r'..\..\data\price_scales.dta')
df_providers = pd.read_stata(r'..\..\data\providers.dta')

#Renaming needed columns
df_prices = df_prices.rename(columns={"price":"price_negotiated"})
df_providers = df_providers.rename(columns={'price':'price_provider','year':'year_negotiated'})

#Merging
df_prices['year_negotiated'] = df_prices.groupby(['provider_id'])['year'].apply(lambda x: x-min(x)) #Add year_negotiated for merging
df_prices = df_prices.merge(df_providers, left_on=['provider_id','year_negotiated'], right_on=['provider_id','year_negotiated'], how='left')
df_prices['price_provider'] = df_prices['price_provider'].fillna(value=1) #Fill the missing values in the providers dataset. I assume no negotiation = pay everything
df = df.merge(df_prices, left_on=['insurer_id','provider_id','year'], right_on=['insurer_id','provider_id','year_negotiated'])

#Cleaning
df['price'] = df['price_negotiated']*df['price_provider']
#Check if people are getting younger over time
df = df.sort_values(['consumer_id','year_negotiated']) #Sort by consumer_id and
df['age_shifted'] = df.groupby('consumer_id')['age'].shift(1) #Make a shifted age column within consumer_id groups
df['age_diff'] = - df['age_shifted'] + df['age'] #Make a difference
df['birth_year'] = df['year_y'] - df['age']
df['birth_year'] = df.groupby('consumer_id')['birth_year'].transform('first') #Fix birth year according to the first data
df['change_age'] = df.groupby('consumer_id')['age_diff'].transform('min') #Indicate if change should be made

df.loc[df.change_age<0, 'age'] = - df.loc[df.change_age<0]['birth_year'] + df.loc[df.change_age<0]['year_y'] #Make the change

# df.groupby(['consumer_id'])['age_diff'].apply(lambda x: x==0 if x>=0 else x==1) #????

#Check if people's gender changes
df['gender_changed'] = df.groupby('consumer_id')['female'].transform('mean') #Prepare to check if gender has changed
df.loc[df.gender_changed > 0.5, 'gender_changed'] = 1 #Change all who had gender over 0.5 on average to female
df.loc[df.gender_changed <= 0.5, 'gender_changed'] = 0 #Change all who had gender under 0.5 on average to male
df['female'] = df['gender_changed']
df = df.drop(columns=['gender_changed', 'age_shifted', 'age_diff','change_age', 'birth_year']) #Drop unnecesary columns used in cleaning

#Create aggregate descriptive statistics
#First batch
tbl = df.groupby('provider_id').agg({'age':'mean','price':['mean','median',lambda x: sum(x)/sum(df['price'])]}) #aggregate

tbl.columns = pd.MultiIndex.from_tuples([
    ('Age', 'Mean'),
    ('Amount paid', 'Mean'),
    ('Amount paid', 'Median'),
    ('Amount paid', 'Market share'),]) #Make nice headers

tbl.index.names = ['Hospital'] #rename index
tbl.to_latex('../../Output/tbl1.tex', multicolumn=True,
encoding='utf-8', escape=False, float_format='%.03f') #Save latex

print(tbl)

#Second batch
tbl = df.groupby('year_y').agg({
    'consumer_id':lambda x: len(x),
    'price':'mean',
    'public':lambda x: sum(x)/len(x)
})

tbl.columns = pd.MultiIndex.from_tuples([
    ('Patients', 'Number'),
    ('Amount paid', 'Mean'),
    ('Public hospitals', 'Share')]) #Make nice headers

tbl.index.names = ['Year'] #rename index
tbl.to_latex('../../Output/tbl2.tex', multicolumn=True,
encoding='utf-8', escape=False, float_format='%.03f') #Save latex

#Export the data
df.to_hdf(r'..\..\data\clean_data.hdf',key='df')

print(tbl)