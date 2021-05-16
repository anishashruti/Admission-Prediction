{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "     Serial No.  GRE Score  TOEFL Score  University Rating  SOP  LOR   CGPA  \\\n",
      "0             1        337          118                  4  4.5   4.5  9.65   \n",
      "1             2        324          107                  4  4.0   4.5  8.87   \n",
      "2             3        316          104                  3  3.0   3.5  8.00   \n",
      "3             4        322          110                  3  3.5   2.5  8.67   \n",
      "4             5        314          103                  2  2.0   3.0  8.21   \n",
      "..          ...        ...          ...                ...  ...   ...   ...   \n",
      "495         496        332          108                  5  4.5   4.0  9.02   \n",
      "496         497        337          117                  5  5.0   5.0  9.87   \n",
      "497         498        330          120                  5  4.5   5.0  9.56   \n",
      "498         499        312          103                  4  4.0   5.0  8.43   \n",
      "499         500        327          113                  4  4.5   4.5  9.04   \n",
      "\n",
      "     Research  Chance of Admit   \n",
      "0           1              0.92  \n",
      "1           1              0.76  \n",
      "2           1              0.72  \n",
      "3           1              0.80  \n",
      "4           0              0.65  \n",
      "..        ...               ...  \n",
      "495         1              0.87  \n",
      "496         1              0.96  \n",
      "497         1              0.93  \n",
      "498         0              0.73  \n",
      "499         0              0.84  \n",
      "\n",
      "[500 rows x 9 columns]\n"
     ]
    }
   ],
   "source": [
    "#importing the libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "df=pd.read_csv('admission.csv')\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Extracting x and y as input and output variable\n",
    "x = df.iloc[:,1:8].values\n",
    "y = df.iloc[:,-1].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "stsc=StandardScaler()\n",
    "x=stsc.fit_transform(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.67519343]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.decomposition import PCA\n",
    "pca = PCA(n_components=1)\n",
    "x = pca.fit_transform(x)\n",
    "var = pca.explained_variance_ratio_\n",
    "print(var)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size = 0.2, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "rfr=RandomForestRegressor(n_estimators=21,random_state=42)\n",
    "rfr.fit(x_tr,y_tr)\n",
    "y_pred=rfr.predict(x_te)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(x_te,y_te,c='red')   \n",
    "plt.plot(x_te,y_pred,c='blue')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.06780476190476191\n",
      "0.00903751700680272\n",
      "0.482528120539789\n",
      "0.09506585615668077\n"
     ]
    }
   ],
   "source": [
    "print(mean_absolute_error(y_te,y_pred))\n",
    "mse = mean_squared_error(y_te,y_pred)\n",
    "print(mse)\n",
    "print(r2_score(y_te,y_pred))\n",
    "print(np.sqrt(mse))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average absolute error: 0.07\n",
      "Accuracy: 89.03 %.\n"
     ]
    }
   ],
   "source": [
    "errors = abs(y_pred -y_te)\n",
    "print('Average absolute error:', round(np.mean(errors), 2))\n",
    "mape = 100 * (errors / y_te)\n",
    "accuracy = 100 - np.mean(mape)\n",
    "print('Accuracy:', round(accuracy, 2), '%.')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
