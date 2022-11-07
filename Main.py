import streamlit as st
import requests
import cv2
import numpy as np
import PIL.Image as Image
import base64

address="http://127.0.0.1:5000/image"

st.markdown("<h1 style='text-align: center; color: white;'>This is Anime Recognition Web</h1>", unsafe_allow_html=True)

### Load fontend
with open("./fontend/type.css") as cssFile:
    st.markdown(f'<style>{cssFile.read()}</style>', unsafe_allow_html=True)

with open("./fontend/main.html") as htmlFile:
    st.markdown(htmlFile.read(), unsafe_allow_html=True)

### Main process
file=st.file_uploader("",type=['jpg', 'png', 'jpeg'])
img_placeholder = st.empty()
if file is not None:
    button=st.button('Recognize')
    if button:
        content = Image.open(file)
        image_np = np.asarray(content)
        _, im_buf_arr  = cv2.imencode(".jpg", image_np)
        content=im_buf_arr.tobytes()
        ## Request
        rp=requests.post(address, files={'file': content})
        if(rp):
            print("Sent Successful")
        # ## From base64 to image
            decode = base64.decodebytes(rp.content)
            im_arr = np.frombuffer(decode, dtype=np.uint8)
            file = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        else:
            print("Failed to Send")
    image=st.image(file)
    st.write("File Uploaded Successfully!")
