Run Train.py to train an lstm on data (located in the Data folder)
Run Test.py to test the saved model
make sure the model name, jump, and look_back match on each of these files

The LSTM is trained to observe 'look_back' number of days and make a prediction for the next day.
While testing The previous predicted price can be used to predic the next day's price also, depending on the 'jump' number

For eg,
while testing:
if look_back=28,
   jump=4
   
day 1 - day 28 is first observation (input) --> day 29(predicted price)
day 2 - day 28(actual price) & day 29(predicted price) --> day 30(predicted price)
day 3 - day 28(actual price) & day 29 - day 30(predicted price) --> day 31(predicted price)
day 4 - day 28(actual price) & day 29 - day 31(predicted price) --> day 32(predicted price)
then the input data is again reset to the actual prices
day 5 - day 32(actual price)  --> day 33(predicted price)
So on...

The first value printed is the mean correlation between predicted and actual prices on days between reset (that happen every 'jump' number of days)
The second vale printed is the overall covariance between predicted and actual prices on all days in the testing data (this will include the resets)