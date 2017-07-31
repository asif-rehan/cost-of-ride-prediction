'''
Created on Jul 31, 2017

@author: Asif Rehan
'''
import pandas
import numpy 
import os
import random
from sklearn import linear_model
from matplotlib import pyplot as plt


def read_data(filepath):
    data = pandas.read_csv(filepath)
    return data

def calc_dist_between_lat_lon_pairs(lat,lon, lat2, lon2):
    '''
    calculates the haversine distance between two GPS points
    
    Ref: http://stackoverflow.com/a/29546836/3012255
    '''
    
    lon1, lat1, lon2, lat2 = map(numpy.radians, [lon, lat, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = numpy.sin(dlat/2.0)**2 + numpy.cos(lat1) * numpy.cos(lat2) * numpy.sin(dlon/2.0)**2

    c = 2 * numpy.arcsin(numpy.sqrt(a))
    km = 6367 * c
    miles = km / 1.61
    return miles

def get_manhattan_distance(data_row):
    '''
    get a middle point. Say this is (lon1, lat2), could also be (lon2, lat1). 
    Go with the first one for now 
    '''
    lat1, lon1 = data_row['Latitude'], data_row['Longitude']
    lat2, lon2 = data_row['AirportLatitude'], data_row['AirportLongitude']
    manhattan_distance1 = calc_dist_between_lat_lon_pairs(lat1, lon1, lat2, lon1)
    manhattan_distance2 = calc_dist_between_lat_lon_pairs(lat2, lon2, lat2, lon1)
    
    total_manhattan_distance = manhattan_distance1 + manhattan_distance2
    
    return  total_manhattan_distance
    
def generate_exploratory_plots(data):
    return None

def main(filepath, training_size=70):
    '''
    steps:
    =====
    1. read data
    2. create a function to get Manhattan distance for a given row ID (starting from 0)
    3. compile the distance for each user to use it a x1 variable, the other x2 variable is the #of passenger
    4. construct the regression model by using randomly selected 70% of the total data
    5. predict the model by pluggin in the remaining 30% of the data
    6. evaluate the model performance by using RMSE, MAE, MAPE values since this is a regression problem
    '''
    data = read_data(filepath)
    
    #apply manhattan distance function and add as a new column to the data frame
    data['distance_mile'] = data.apply(lambda x: get_manhattan_distance(x), axis=1)
    generate_exploratory_plots(data)
    
    #split the given data into training and testing dataset
    training_set_index = random.sample(data.index.tolist(), int(training_size/100.0*data.shape[0]))
    train_data = data[data.index.isin(training_set_index)]
    test_data = data[~data.index.isin(training_set_index)]
    
    #split into X-train
    X_train = train_data[['PayingPax', 'distance_mile']]
    Y_train = train_data['Fare']
    
    #split into X-test
    X_test = test_data[['PayingPax', 'distance_mile']]
    Y_test = test_data['Fare']
    
    #build regression model
    regr_model = linear_model.LinearRegression()
    regr_model.fit(X_train, Y_train)
    
    Y_hat = regr_model.predict(X_test)
    print 'Coefficients: \n', regr_model.coef_
    
    # The mean squared error
    print "Mean squared error: %.2f" % numpy.mean(Y_hat - Y_test) ** 2
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr_model.score(X_test, Y_test))
    
    # Plot outputs
    plt.scatter(X_test, Y_test,  color='black')
    plt.plot(X_test, Y_hat, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    
    plt.show()
    
    return X_train

if __name__ == '__main__':
    main(os.path.join('./../data/example_problem.csv'))