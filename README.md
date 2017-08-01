Variables
---------
X-var = [PayingPax, Distance_mile]
Y-var = [Fare]

Coefficients
------------
[ 15.94881733,   0.65643556]

Final Model
----------- 
Y = 15.95 * X1 + 0.66 * X2

i.e. Fare = 15.95 * PayingPax + 0.66 * Distance_mile

Performance metrics
-------------------
R-squared value: 0.92

Mean absolute error (MAE) : $2.17

Mean squared error (MSE): $10.17

Root mean squared error (RMSE): $3.19

Mean absolute percentage error (MAPE): 8.99 %

The coefficient of determination (R-squared value=0.92) tells that the model explains 92% of the variability of the data, which is very good. Similarly, MAPE is below 9%. The RMSE is only $3.19, meaning the predicted fare in the test dataset was only a little off.  RMSE can be likened to the Std Dev of the error.
