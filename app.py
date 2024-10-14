import streamlit as st
import numpy as np
import os
import cv2
import pickle
from PIL import Image
# from skimage import color

mapper = {0:'angry',
 1:'disgust',
 2:'fear',
 3:'happy',
 4:'neutral',
 5:'sad',
 6:'surprise'}

def main():
	st.title("EMOTION DETECTOR")
	st.text("Upload your image to predict your emotion")
	my_model=pickle.load(open('face_emotion.p','rb'))
	upload_file=st.file_uploader("Browse Image",type=["jpg",'png','jpeg'])
	if upload_file is not None:
		img=Image.open(upload_file)
		st.image(img,caption="Uploaded Image")
	if st.button("PREDICT"):
		st.write("Result....")
		img=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2GRAY)
		img=cv2.resize(img,(48,48))
		img = np.stack((img,)*3, axis=-1)

		# test_img=Image.open(url).convert('L')

		# test_prediction=my_model.predict(test_img)
		# test_prediction=np.argmax(test_prediction,axis=1)
		test_prediction=mapper[np.argmax(my_model.predict(img.reshape(1,48,48,3)))]
		st.write(f'PREDICTED OUTPUT : {test_prediction}')

if __name__ == "__main__":
    main()
