base - conda
python version - 3.9 
******************************************************************************************************************************************************
Python Modules -

os - provides a way to interact with the operating system. 
PIL - for working with images.
glob - provides a way to retrieve files and directories.
cv2 - providing various functions and classes for image and video processing and analysis.
labelme - for creating and editing annotations for computer vision.
albumentations - for augmenting dataset.
numpy -  for numerical computing.
json - provides functions for encoding and decoding JSON(labels).
tensorflow - providing a comprehensive ecosystem for building, training, and deploying machine learning models.
matplotlib - provides a wide range of tools for generating plots
scipy - provides a collection of mathematical algorithms and functions for optimization, integration, interpolation, signal processing.
random - for generating random numbers from different distributions.
huggingface_hub - provides a central repository for sharing and reusing models.
keras - for building and training deep learning models.


Use !pip install module_name to resolve any missing dependencies.

********************************************************************************************************************************************************
Dataset - {link to drive}

data - It contains human faces and their forheads as labels.
aug_data - It contains augmented data.
lol_dataset - It contains low light and high light images for training of Zero_dce model.

********************************************************************************************************************************************************
Models - 
forehead.h5 - Forehead detection Model for automated ROI detection and motion handling tool.
model.h5 - Zero dce model trained on updated lol dataset.
haarcascade_frontalface_default - Hugging_face model that give facial landmarks using which we can try to predict forehead.

********************************************************************************************************************************************************
Notebooks - 


1)- Initial_data_processing_for_dataset_creation - This notebook contains code that we have used for initial preprocessing and augmentation of dataset.

2)- Forehead_detection_model - This notebook contains code for training our forehead detections model.

3)- Prediction_and_performance_of_our_model - This notebook contains code that is used for doing prediction and evaluating performance of our forehead_detection model.

4)- Zero_dce - This notebook contains code for training of zero dce model on updated lol dataset.

5)- low_light - This notebook contains code for different models that we have used for low light image enhancement.

6)- heart_rate_prediction - This notebook contains code for heart rate predcition.


All notebooks contains detailed explanation of code Written.

***********************************************************************************************************************************************************

**Note - Keep all notebooks, data , aug_data , lol_dataset , models in same directory(Folder).**

Please feel free to contact us in case of any issue.

Allan Robey - allanrobey22@iitk.ac.in
Avnish Tripathi - avnisht22@iitk.ac.in
Divyesh Tripathi- divyeshdt22@iitk.ac.in
Kush Shah - kushshah22@iitk.ac.in
Pulkit Sharma - pulkitsh22@iitk.ac.in

