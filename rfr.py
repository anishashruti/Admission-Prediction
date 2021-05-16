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
    "#importing libraries and dataset\n",
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
   "outputs": [],
   "source": [
    "x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=0)\n",
    "rfr=RandomForestRegressor(n_estimators=21,random_state=42)\n",
    "rfr.fit(x_tr,y_tr)\n",
    "y_pred=rfr.predict(x_te)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.04543809523809524\n",
      "0.004417668934240363\n",
      "0.7470522661795695\n",
      "0.06646554697164812\n"
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
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[  1.   337.   118.   ...   4.5    9.65   1.  ]\n",
      " [  1.   324.   107.   ...   4.5    8.87   1.  ]\n",
      " [  1.   316.   104.   ...   3.5    8.     1.  ]\n",
      " ...\n",
      " [  1.   330.   120.   ...   5.     9.56   1.  ]\n",
      " [  1.   312.   103.   ...   5.     8.43   0.  ]\n",
      " [  1.   327.   113.   ...   4.5    9.04   0.  ]]\n"
     ]
    }
   ],
   "source": [
    "x_copy = x[:,:]\n",
    "const = np.ones((500,1)).astype(int)\n",
    "x_copy = np.append(arr = const, values=x_copy, axis=1)\n",
    "print(x_copy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_opt = np.array(x_copy[:,[0,1,2,3,4,5,6,7]],dtype = float)"
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
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:                      y   R-squared:                       0.822\n",
      "Model:                            OLS   Adj. R-squared:                  0.819\n",
      "Method:                 Least Squares   F-statistic:                     324.4\n",
      "Date:                Thu, 24 Sep 2020   Prob (F-statistic):          8.21e-180\n",
      "Time:                        23:03:57   Log-Likelihood:                 701.38\n",
      "No. Observations:                 500   AIC:                            -1387.\n",
      "Df Residuals:                     492   BIC:                            -1353.\n",
      "Df Model:                           7                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "const         -1.2757      0.104    -12.232      0.000      -1.481      -1.071\n",
      "x1             0.0019      0.001      3.700      0.000       0.001       0.003\n",
      "x2             0.0028      0.001      3.184      0.002       0.001       0.004\n",
      "x3             0.0059      0.004      1.563      0.119      -0.002       0.013\n",
      "x4             0.0016      0.005      0.348      0.728      -0.007       0.011\n",
      "x5             0.0169      0.004      4.074      0.000       0.009       0.025\n",
      "x6             0.1184      0.010     12.198      0.000       0.099       0.137\n",
      "x7             0.0243      0.007      3.680      0.000       0.011       0.037\n",
      "==============================================================================\n",
      "Omnibus:                      112.770   Durbin-Watson:                   0.796\n",
      "Prob(Omnibus):                  0.000   Jarque-Bera (JB):              262.104\n",
      "Skew:                          -1.160   Prob(JB):                     1.22e-57\n",
      "Kurtosis:                       5.684   Cond. No.                     1.30e+04\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 1.3e+04. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    }
   ],
   "source": [
    "import statsmodels.api as sm\n",
    "p_value=sm.OLS(endog=y,exog=x_opt).fit()\n",
    "print(p_value.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.822</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.820</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   379.1</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Thu, 24 Sep 2020</td> <th>  Prob (F-statistic):</th> <td>4.29e-181</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>23:03:57</td>     <th>  Log-Likelihood:    </th> <td>  701.32</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>   500</td>      <th>  AIC:               </th> <td>  -1389.</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>   493</td>      <th>  BIC:               </th> <td>  -1359.</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     6</td>      <th>                     </th>     <td> </td>    \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>const</th> <td>   -1.2800</td> <td>    0.103</td> <td>  -12.371</td> <td> 0.000</td> <td>   -1.483</td> <td>   -1.077</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>x1</th>    <td>    0.0019</td> <td>    0.001</td> <td>    3.694</td> <td> 0.000</td> <td>    0.001</td> <td>    0.003</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>x2</th>    <td>    0.0028</td> <td>    0.001</td> <td>    3.236</td> <td> 0.001</td> <td>    0.001</td> <td>    0.005</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>x3</th>    <td>    0.0064</td> <td>    0.004</td> <td>    1.820</td> <td> 0.069</td> <td>   -0.001</td> <td>    0.013</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>x4</th>    <td>    0.0173</td> <td>    0.004</td> <td>    4.380</td> <td> 0.000</td> <td>    0.010</td> <td>    0.025</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>x5</th>    <td>    0.1190</td> <td>    0.010</td> <td>   12.481</td> <td> 0.000</td> <td>    0.100</td> <td>    0.138</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>x6</th>    <td>    0.0244</td> <td>    0.007</td> <td>    3.691</td> <td> 0.000</td> <td>    0.011</td> <td>    0.037</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td>111.782</td> <th>  Durbin-Watson:     </th> <td>   0.800</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 258.656</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td>-1.152</td>  <th>  Prob(JB):          </th> <td>6.82e-57</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 5.667</td>  <th>  Cond. No.          </th> <td>1.29e+04</td>\n",
       "</tr>\n",
       "</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.29e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:                      y   R-squared:                       0.822\n",
       "Model:                            OLS   Adj. R-squared:                  0.820\n",
       "Method:                 Least Squares   F-statistic:                     379.1\n",
       "Date:                Thu, 24 Sep 2020   Prob (F-statistic):          4.29e-181\n",
       "Time:                        23:03:57   Log-Likelihood:                 701.32\n",
       "No. Observations:                 500   AIC:                            -1389.\n",
       "Df Residuals:                     493   BIC:                            -1359.\n",
       "Df Model:                           6                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "==============================================================================\n",
       "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------\n",
       "const         -1.2800      0.103    -12.371      0.000      -1.483      -1.077\n",
       "x1             0.0019      0.001      3.694      0.000       0.001       0.003\n",
       "x2             0.0028      0.001      3.236      0.001       0.001       0.005\n",
       "x3             0.0064      0.004      1.820      0.069      -0.001       0.013\n",
       "x4             0.0173      0.004      4.380      0.000       0.010       0.025\n",
       "x5             0.1190      0.010     12.481      0.000       0.100       0.138\n",
       "x6             0.0244      0.007      3.691      0.000       0.011       0.037\n",
       "==============================================================================\n",
       "Omnibus:                      111.782   Durbin-Watson:                   0.800\n",
       "Prob(Omnibus):                  0.000   Jarque-Bera (JB):              258.656\n",
       "Skew:                          -1.152   Prob(JB):                     6.82e-57\n",
       "Kurtosis:                       5.667   Cond. No.                     1.29e+04\n",
       "==============================================================================\n",
       "\n",
       "Notes:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "[2] The condition number is large, 1.29e+04. This might indicate that there are\n",
       "strong multicollinearity or other numerical problems.\n",
       "\"\"\""
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_opt = np.array(x_copy[:,[0,1,2,3,5,6,7]],dtype = float)\n",
    "p_val = sm.OLS(endog= y,exog = x_opt).fit()\n",
    "p_val.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.821</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.819</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   452.1</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Thu, 24 Sep 2020</td> <th>  Prob (F-statistic):</th> <td>9.97e-182</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>23:03:57</td>     <th>  Log-Likelihood:    </th> <td>  699.65</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>   500</td>      <th>  AIC:               </th> <td>  -1387.</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>   494</td>      <th>  BIC:               </th> <td>  -1362.</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>    \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>const</th> <td>   -1.3357</td> <td>    0.099</td> <td>  -13.482</td> <td> 0.000</td> <td>   -1.530</td> <td>   -1.141</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>x1</th>    <td>    0.0019</td> <td>    0.001</td> <td>    3.760</td> <td> 0.000</td> <td>    0.001</td> <td>    0.003</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>x2</th>    <td>    0.0030</td> <td>    0.001</td> <td>    3.501</td> <td> 0.001</td> <td>    0.001</td> <td>    0.005</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>x3</th>    <td>    0.0193</td> <td>    0.004</td> <td>    5.092</td> <td> 0.000</td> <td>    0.012</td> <td>    0.027</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>x4</th>    <td>    0.1230</td> <td>    0.009</td> <td>   13.221</td> <td> 0.000</td> <td>    0.105</td> <td>    0.141</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>x5</th>    <td>    0.0252</td> <td>    0.007</td> <td>    3.814</td> <td> 0.000</td> <td>    0.012</td> <td>    0.038</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td>109.027</td> <th>  Durbin-Watson:     </th> <td>   0.800</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 248.874</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td>-1.130</td>  <th>  Prob(JB):          </th> <td>9.07e-55</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 5.615</td>  <th>  Cond. No.          </th> <td>1.23e+04</td>\n",
       "</tr>\n",
       "</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.23e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:                      y   R-squared:                       0.821\n",
       "Model:                            OLS   Adj. R-squared:                  0.819\n",
       "Method:                 Least Squares   F-statistic:                     452.1\n",
       "Date:                Thu, 24 Sep 2020   Prob (F-statistic):          9.97e-182\n",
       "Time:                        23:03:57   Log-Likelihood:                 699.65\n",
       "No. Observations:                 500   AIC:                            -1387.\n",
       "Df Residuals:                     494   BIC:                            -1362.\n",
       "Df Model:                           5                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "==============================================================================\n",
       "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------\n",
       "const         -1.3357      0.099    -13.482      0.000      -1.530      -1.141\n",
       "x1             0.0019      0.001      3.760      0.000       0.001       0.003\n",
       "x2             0.0030      0.001      3.501      0.001       0.001       0.005\n",
       "x3             0.0193      0.004      5.092      0.000       0.012       0.027\n",
       "x4             0.1230      0.009     13.221      0.000       0.105       0.141\n",
       "x5             0.0252      0.007      3.814      0.000       0.012       0.038\n",
       "==============================================================================\n",
       "Omnibus:                      109.027   Durbin-Watson:                   0.800\n",
       "Prob(Omnibus):                  0.000   Jarque-Bera (JB):              248.874\n",
       "Skew:                          -1.130   Prob(JB):                     9.07e-55\n",
       "Kurtosis:                       5.615   Cond. No.                     1.23e+04\n",
       "==============================================================================\n",
       "\n",
       "Notes:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "[2] The condition number is large, 1.23e+04. This might indicate that there are\n",
       "strong multicollinearity or other numerical problems.\n",
       "\"\"\""
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_opt = np.array(x_copy[:,[0,1,2,5,6,7]],dtype = float)\n",
    "p_val = sm.OLS(endog= y,exog = x_opt).fit()\n",
    "p_val.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test,y_train,y_test = train_test_split(x_opt,y,test_size = 0.2,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "opt_rfr = RandomForestRegressor(n_estimators=21,random_state=42)\n",
    "opt_rfr.fit(x_train,y_train)\n",
    "y_pred=opt_rfr.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.04771904761904759\n",
      "0.004811730158730157\n",
      "0.764707571700237\n",
      "0.06936663577491817\n"
     ]
    }
   ],
   "source": [
    "print(mean_absolute_error(y_test,y_pred))\n",
    "mse = mean_squared_error(y_test,y_pred)\n",
    "print(mse)\n",
    "print(r2_score(y_test,y_pred))\n",
    "print(np.sqrt(mse))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average absolute error: 0.05\n",
      "Accuracy: 92.36 %.\n"
     ]
    }
   ],
   "source": [
    "errors = abs(y_pred -y_test)\n",
    "print('Average absolute error:', round(np.mean(errors), 2))\n",
    "mape = 100 * (errors / y_test)\n",
    "accuracy = 100 - np.mean(mape)\n",
    "print('Accuracy:', round(accuracy, 2), '%.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
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
    "plt.scatter(x_test[:,4],y_test,c='blue')\n",
    "plt.plot(x_test[:,4],y_pred,c='green')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
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
    "plt.scatter(x_test[:,4],y_pred,c='blue')\n",
    "plt.xlabel('CGPA')\n",
    "plt.ylabel('Chances')\n",
    "plt.title('CGPA vs Chances')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
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
    "plt.scatter(x_test[:,5],y_pred,c='blue')\n",
    "plt.xlabel('Research')\n",
    "plt.ylabel('Chances')\n",
    "plt.title('Research vs Chances')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
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
    "plt.scatter(x_te[:,3],y_pred,c='blue')\n",
    "plt.xlabel('LOR')\n",
    "plt.ylabel('Chances')\n",
    "plt.title('LOR vs Chances')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
