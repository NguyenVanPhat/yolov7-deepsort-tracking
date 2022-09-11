import streamlit as st
import wget
from detection_helpers import *
# from tracking_helpers import *
# from bridge_wrapper import *
# from PIL import Image

st.set_page_config(
    page_title="Web App of Phat",
    page_icon="💽",
)
st.markdown("<h1 style='text-align: center; color: red;'>🎥 Get Video Random 📀</h1>", unsafe_allow_html=True)
st.header('')
st.header('')
path = ""
# os.system("wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt")
wget.download("https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt")
