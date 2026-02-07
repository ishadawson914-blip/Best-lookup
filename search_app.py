import streamlit as st
import pandas as pd
import urllib.parse
import numpy as np
import re
import cv2
from PIL import Image
from thefuzz import process, fuzz
from streamlit_cropper import st_cropper

# --- OCR Setup ---
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

@st.cache_resource
def load_ocr():
    if OCR_AVAILABLE:
        return easyocr.Reader(['en'], gpu=False)
    return None

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv('SortCart.csv')
    unique_streets = sorted(df['StreetName'].unique())
    return df, unique_streets

SUFFIX_MAP = {'ST': 'STREET', 'RD': 'ROAD', 'AVE': 'AVENUE', 'DR': 'DRIVE', 'PL': 'PLACE'}

def normalize_text(text):
    if not text: return ""
    words = text.upper().split()
    return " ".join([SUFFIX_MAP.get(w, w) for w in words])

df, street_list = load_data()
reader = load_ocr()

# --- Initialize Session State (This fixes the "Second Test" error) ---
if 'scanned_street' not in st.session_state:
    st.session_state.scanned_street = None
if 'scanned_no' not in st.session_state:
    st.session_state.scanned_no = None

def make_map_link(row, specific_no=None):
    num = specific_no if specific_no else row['StreetNoMin']
    address = f"{num} {row['StreetName']}, {row['Suburb']}, NSW {row['Postcode']}, Australia"
    query = urllib.parse.quote(address)
    return f"https://www.google.com/maps/search/?api=1&query={query}"

st.set_page_config(page_title="Leightonfield Sorting", layout="wide")
st.title("📦 Leightonfield Address Sorter")

# --- Sidebar ---
option = st.sidebar.selectbox("Search by:", ["Street Address", "Beat Number", "Suburb"])
if st.sidebar.button("Clear All Data"):
    st.session_state.scanned_street = None
    st.session_state.scanned_no = None
    st.rerun()

# --- OCR & Cropping ---
if option == "Street Address" and OCR_AVAILABLE:
    with st.expander("📸 Scan Address Label", expanded=True):
        img_file = st.camera_input("Take a photo")
        
        if img_file:
            img = Image.open(img_file)
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
            
            if st.button("🚀 Process Scanned Address"):
                img_np = np.array(cropped_img)
                with st.spinner("Analyzing..."):
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    
                    results_ocr = reader.readtext(enhanced, paragraph=False, contrast_ths=0.05)
                    detected_strings = [normalize_text(res[1]) for res in results_ocr]
                    full_blob = " ".join(detected_strings)
                    
                    best_match, score = process.extractOne(full_blob, street_list, scorer=fuzz.partial_ratio)
                    
                    if score > 60:
                        st.session_state.scanned_street = best_match
                        # Extract number
                        nums = re.findall(r'\b(\d+)\b', full_blob)
                        potential = [n for n in nums if not (len(n) == 4 and n.startswith('2'))]
                        st.session_state.scanned_no = potential[-1] if potential else (nums[-1] if nums else None)
                        st.success(f"Match Found: {st.session_state.scanned_street}")
                    else:
                        st.error("No clear match. Try again.")

# --- Search UI ---
results_df = pd.DataFrame()
searched_no = None

if option == "Street Address":
    col1, col2 = st.columns([3, 1])
    with col1:
        st_name = st.selectbox("Street Name", options=street_list, 
                               index=street_list.index(st.session_state.scanned_street) if st.session_state.scanned_street in street_list else None)
    with col2:
        st_no_str = st.text_input("Number", value=st.session_state.scanned_no if st.session_state.scanned_no else "")
        if st_no_str.isdigit(): searched_no = int(st_no_str)
   
    if st_name:
        if searched_no is not None:
            parity = 2 if searched_no % 2 == 0 else 1
            results_df = df[(df['StreetName'] == st_name) & (df['EvenOdd'] == parity) & (df['StreetNoMin'] <= searched_no) & (df['StreetNoMax'] >= searched_no)].copy()
        else:
            results_df = df[df['StreetName'] == st_name].copy()

# --- Results ---
if not results_df.empty:
    primary_beat = results_df.iloc[0]['BeatNo']
    # Fixed Markdown Card
    st.markdown(f"""
        <div style="background-color:#007BFF;padding:20px;border-radius:10px;text-align:center;">
            <h1 style="color:white;margin:0;font-size:50px;">BEAT: {primary_beat}</h1>
        </div>
    """, unsafe_content_allowed=True)
    
    results_df['Map Link'] = results_df.apply(lambda row: make_map_link(row, searched_no), axis=1)
    st.dataframe(results_df[['Map Link', 'BeatNo', 'Suburb', 'StreetName', 'StreetNoMin', 'StreetNoMax', 'TeamNo']], 
                 column_config={"Map Link": st.column_config.LinkColumn("Maps", display_text="📍 View")}, hide_index=True)



































