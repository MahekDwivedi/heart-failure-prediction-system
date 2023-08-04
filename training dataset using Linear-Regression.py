# Training and testing data 

# USING LINEAR REGRESSION
from sklearn.model_selection import train_test_split

x=df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]
np.random.seed(42)

# Split into train & test set
x_train, x_test, y_train, y_test = train_test_split( x, y ,test_size=0.3)

from sklearn.linear_model import LinearRegression
mymodel= LinearRegression().fit(x_train , y_train)        # forms best fit line
linear_predictions= mymodel.predict(x_test)              

linear_predictions

plt.plot(linear_predictions ,color='blue')
plt.plot( y_test,color='red')
plt.xlim(0,200)
plt.legend(['predictions' , 'testdata'])


test_residual = y_test - linear_predictions          #error residual
test_residual
