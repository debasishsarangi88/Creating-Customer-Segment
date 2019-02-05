# Creating-Customer-Segment
Unsupervised ML Project for Udacity ML Engineer Nanodegree

Project Description
This is a project for the Machine Learning Engineer Nanodegree. In this project I apply unsupervised learning techniques on product spending data collected for customers of a wholesale distributor to identify customer segments hidden in the data.

Here, I first explore the data to determine if any product categories highly correlate with one another by observing a small subset of the data and also by ploting a scater matrix. Afterwards, I preprocess the data by scaling each product category and then identifying (and removing) unwanted outliers. Then I apply PCA transformations to the data and implement a clustering algorithm (Gaussian Mixture Model)to segment the transformed customer data. Finally, I compare the segmentation found with an additional labeling and consider ways this information could assist the wholesale distributor with future service changes.

Install
This project requires Python 2.7 and the following Python libraries installed:

NumPy
Pandas
matplotlib
scikit-learn
You will also need to have software installed to run and execute a Jupyter Notebook

If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer.

Code
The main code for this project is located in the customer_segments.ipynb notebook file. 

Run
In a terminal or command window, navigate to the top-level project directory customer_segments/ (that contains this README) and run one of the following commands:

ipython notebook customer_segments.ipynb
or

jupyter notebook customer_segments.ipynb
This will open the Jupyter Notebook software and project file in your browser.

Data
The customer segments data is included as a selection of 440 data points collected on data found from clients of a wholesale distributor in Lisbon, Portugal. More information can be found on the UCI Machine Learning Repository.

Note (m.u.) is shorthand for monetary units.

Features

Fresh: annual spending (m.u.) on fresh products (Continuous);

Milk: annual spending (m.u.) on milk products (Continuous);

Grocery: annual spending (m.u.) on grocery products (Continuous);

Frozen: annual spending (m.u.) on frozen products (Continuous);

Detergents_Paper: annual spending (m.u.) on detergents and paper products (Continuous);

Delicatessen: annual spending (m.u.) on and delicatessen products (Continuous);

Channel: {Hotel/Restaurant/Cafe - 1, Retail - 2} (Nominal)

Region: {Lisnon - 1, Oporto - 2, or Other - 3} (Nominal)
