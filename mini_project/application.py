'''
Created on Jul 31, 2017

@author: Asif Rehan
'''
import pandas
import numpy 
import os
import random
from sklearn import linear_model, metrics
from sklearn import model_selection
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import mstats



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
    
def generate_exploratory_plots(data, img_id=0):
    for x, y in [('PayingPax', 'Distance_mile'), ('PayingPax', 'Fare'), ('Distance_mile', 'Fare')]:
        plt.scatter(data[x], data[y])
        plt.xlabel(x)
        plt.ylabel(y)
        img_title = y + ' Vs ' + x
        plt.title(img_title)
        img_filename = os.path.join('./../output/{}_{}.png'.format(img_title, img_id))
        
        if os.path.isfile(img_filename):
            os.remove(img_filename)
        plt.savefig(img_filename)    


def show_diagnostics(X_test, Y_test, regr_model, Y_hat):
    #show model params ouputs
    print 'Coefficients: \n', regr_model.coef_
        
    # The mean squared error
    print "Mean absolute error: %.2f" % metrics.mean_absolute_error(Y_test, Y_hat)
    print "Mean squared error: %.2f" % metrics.mean_squared_error(Y_test, Y_hat)
    print "R-squared value: %.2f" % metrics.r2_score(Y_test, Y_hat)
    #print "Median absolute error: %.2f" % metrics.median_absolute_error(Y_test, Y_hat)
    

def plot_model_outputs(X_test, Y_test, Y_hat):
    plt.scatter(Y_test, Y_hat, color='black')
    plt.xlabel('Y_test')
    plt.ylabel('Y_hat')
    img_title = 'Y_hat Vs Y_test'
    plt.title(img_title)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(os.path.join('./../output/output_{}.png'.format(img_title)))
    
    #plt.scatter(X_test['PayingPax'], X_test['Distance_mile'], Y_hat, color='blue', linewidth=3)

def main(filepath, data_filter_percentile=[0.05, 0.05], test_size=0.3):
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
    data['Distance_mile'] = data.apply(lambda x: get_manhattan_distance(x), axis=1)
    
    
    #generate exploratory plots
    generate_exploratory_plots(data, img_id='raw')
    
    #identify and remove bad data
    min_quantile = data.quantile(0.05)
    max_quantile = data.quantile(0.95)
    data = data[( data['Distance_mile'] > min_quantile['Distance_mile'] ) &
                ( data['Distance_mile'] < max_quantile['Distance_mile'] ) ] 
    
    #data = data[(numpy.abs(stats.zscore(data)) < 3).all(axis=1)]
    
    #generate exploratory plots
    generate_exploratory_plots(data, img_id='after filtering top and bottom 5pct')
    
    #split the given data into training and testing dataset
    X = data[['PayingPax', 'Distance_mile']]
    Y = data['Fare']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=123)
    
    #build regression model
    regr_model = linear_model.LinearRegression(normalize=True)
    regr_model.fit(X_train, Y_train)
    
    #predict on test data set
    Y_hat = regr_model.predict(X_test)
    
    #show model performance metrics
    show_diagnostics(X_test, Y_test, regr_model, Y_hat)
    
    # Plot outputs
    plot_model_outputs(X_test, Y_test, Y_hat)
    
    return None

if __name__ == '__main__':
    data_filepath = os.path.join('./../data/example_problem.csv')
    main(data_filepath)