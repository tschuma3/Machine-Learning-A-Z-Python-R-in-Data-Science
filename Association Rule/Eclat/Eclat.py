import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

#Data Preprocessing 
dataset = pd.read_csv(r'Association Rule\Eclat\Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(1, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

#Training the Apriori model on the dataset
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_lenght=2)

#region Visualizing the results

#Displaying the first results coming directly from the output of the apriori function
results =  list(rules)
print(results)

#Putting the results well organized into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

#Displaying the results sorted by decending lifts
print(resultsinDataFrame.nlargest(n = 10, columns = 'Support'))

#endregion