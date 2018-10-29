######################################################################
#						  		     #
# (c) Marko Borazio, Embedded Sensing Systems, TU-Darmstadt, Germany #
# 		   	www.ess.tu-darmstadt.de	     		     #
#						     		     #
######################################################################

Data set website: http://www.ess.tu-darmstadt.de/ichi2014

For detailed information have a look at the following paper:

Marko Borazio, Eugen Berlin, Nagihan Kücükyildiz, Philipp M. Scholl 
and Kristof Van Laerhoven:
"Towards a Benchmark for Wearable Sleep Analysis with Inertial Wrist-worn 
Sensing Units", ICHI 2014, Verona, Italy, IEEE Press, 2014


IMPORTANT NOTE
--------------

This data set is opened up to anyone interested in activity recognition to 
encourage reproducible results. Please cite our paper if you publish results 
on this data set, and consider making your own data sets open for anyone to 
download in a similar fashion. We would also be very interested to hear back 
from you if you use our data in any way and are happy to answer any questions 
or address any remarks related to it. 


PATIENTS INFORMATION
--------------------

The file patients_info.pdf contains information about the patients, e.g., 
patient ID, age or sleep related disorder. The same information is accessable
via the numpy file pat_inf.npy:

>>> import numpy as np
>>> p_dta = np.load('data/pat_inf.npy')

Note: The first field of the list contains the column headers.

!The data set obtained from the sleeping lab was already anonymized.!


DATA SET CONTENTS
-----------------

This data set contains high-frequent (100Hz) data recorded with a wrist-worn 
data logger from 42 sleeping lab patients along with their data from clinical 
polysomnography (in Python npy format).
The data set consists of the following fields: 
1) 't' = timestamp
2) 'd' = runlength encoding, i.e., how often did the sensor values occur
3) 'x','y','z' = 3D accelerometer values
4) 'l' = light sensor values
5) 'gt' = ground truth, i.e., sleep phases as determined by polysomnography
   --> 0 = unknown, 1-3 = sleep stages 3-1, 5 = REM, 6 = awake, 7 = movement


LOADING DATA
------------

To load the data we need to import the numpy library and issue the load command:

>>> import numpy as np
>>> dta = np.load('data/p002.npy').view(np.recarray)

The 'dta' variable now holds a record array with sensor specific information 
as well as the ground truth in column dta.gt.


DISPLAY THE DATA
----------------

You can display the sensor data along with the ground truth by using:

./show_data.py 'patient_ID'

All available patient IDs are in the patients_info.pdf or in show_data.py 
(function get_userlist).


------------------------------------------------------
For further information do not hesitate to contact us!
------------------------------------------------------
