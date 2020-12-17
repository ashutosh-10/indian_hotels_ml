# Importing modules
#------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#------------------------------------------------------------------

#filter dataset according to user needs
def filter_dataset(city, isBreakfast, isWifi, hasSwimmingpool, rooms, price):
	#read csv file using dataframes
	hotels_df = pd.read_csv("cities.csv", usecols=['CityName','HotelName', 'RoomRent', 'StarRating', 'HotelAddress', 'HotelDescription', 'FreeWifi', 'FreeBreakfast', 'HotelCapacity', 'HasSwimmingPool'], encoding = "ISO-8859-1")
	
	#filter dataset
	filter_city_df = hotels_df[hotels_df["CityName"] == city]
	filter_city_df = filter_city_df[filter_city_df["FreeBreakfast"] == isBreakfast]
	filter_city_df = filter_city_df[filter_city_df["FreeWifi"] == isWifi]
	filter_city_df = filter_city_df[filter_city_df["HasSwimmingPool"] == hasSwimmingpool]
	filter_city_df = filter_city_df[filter_city_df["HotelCapacity"] >= rooms]
	filter_city_df = filter_city_df[filter_city_df["RoomRent"] <= price]

	#if the entered details are not found
	if filter_city_df.shape == (0, 10):
		return

	return filter_city_df

#combines the detals of hotels in single string
def combine_feature(row):
    return str(row['RoomRent'])+' '+str(row['StarRating'])+' '+row['HotelAddress']+' '+row['HotelDescription']+' '+str(row['HotelCapacity'])

# get hotel name from the index value
def get_hotelName(filter_city_df,index):
    return filter_city_df.iloc[index]['HotelName']

# get prices
def get_price(filter_city_df,index):
    return 'Rs. '+str(filter_city_df.iloc[index]['RoomRent'])

#get hotel address
def get_hotelAddress(filter_city_df,index):
    return filter_city_df.iloc[index]['HotelAddress']

#get hotel ratings
def get_hotelRatings(filter_city_df,index):
    return str(filter_city_df.iloc[index]['StarRating'])

#get hotel ratings
def get_hotelDes(filter_city_df,index):
    return filter_city_df.iloc[index]['HotelDescription']

#preprocess data
def data_preprocesing(filter_city_df):
	#remove duplicate hotels
	filter_city_df.drop_duplicates(subset ="HotelName", 
                     keep = 'first', inplace = True)
	#sort the hotels according to their start ratings
	filter_city_df.sort_values("StarRating", inplace = True, ascending=False)

	#combinig the hotel details
	filter_city_df['CombineFeatures'] = filter_city_df.apply(combine_feature, axis=1)
	return filter_city_df

#finding cosine similarties
def cosine_sim(filter_city_df):
	#object of CountVectorizer
	cv = CountVectorizer()

	#count all the vectors from text
	count_matrix = cv.fit_transform(filter_city_df['CombineFeatures'])

	#find cosine similarities
	cos_similarity = cosine_similarity(count_matrix)

	#recommned hotel according to first hotel
	sim_hotels = list(enumerate(cos_similarity[0]))

	#sort hotels
	sim_hotels = sorted(sim_hotels, key=lambda x:x[1], reverse=True)

	return sim_hotels

def recommendation(city, isBreakfast, isWifi, hasSwimmingpool, price, rooms): #input first comes here

	if rooms <= 0:
		raise TypeError

	#filter dataset
	df = filter_dataset(city, isBreakfast, isWifi, hasSwimmingpool, rooms, price)

	#pre_processing dataset
	df = data_preprocesing(df)

	#recommend hotels
	rec_hotels = cosine_sim(df)

	result_hotels = []
	result_price = []
	result_add = []
	result_des = []
	result_rating = []
	count = 0
	for hotel in rec_hotels:
		if count <15:
			result_hotels.append("{}. ".format(count+1)+get_hotelName(df, hotel[0]))
			result_price.append(get_price(df, hotel[0]))
			result_add.append("    "+get_hotelAddress(df, hotel[0]))
			result_des.append("    "+get_hotelDes(df, hotel[0]))
			result_rating.append(get_hotelRatings(df, hotel[0]))
			count += 1

	return result_hotels,result_rating,result_price, result_add, result_des
