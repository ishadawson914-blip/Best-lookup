import streamlit as st
import pandas as pd
import urllib.parse
import numpy as np
import re
import cv2
from PIL import Image
from thefuzz import process, fuzz
from streamlit_cropper import st_cropper

# We use easyocr for the address reading
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Initialize OCR reader
@st.cache_resource
def load_ocr():
    if OCR_AVAILABLE:
        return easyocr.Reader(['en'], gpu=False)
    return None

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('SortCart.csv')
    unique_streets = sorted(df['StreetName'].unique())
    return df, unique_streets

# Helper for normalization
SUFFIX_MAP = {'ST': 'STREET', 'RD': 'ROAD', 'AVE': 'AVENUE', 'DR': 'DRIVE', 'PL': 'PLACE', 'CT': 'COURT'}

def normalize_text(text):
    words = text.upper().split()
    return " ".join([SUFFIX_MAP.get(w, w) for w in words])

df, street_list = load_data()
reader = load_ocr()

def make_map_link(row, specific_no=None):
    num = specific_no if specific_no else row['StreetNoMin']
    address = f"{num} {row['StreetName']}, {row['Suburb']}, NSW {row['Postcode']}, Australia"
    query = urllib.parse.quote(address)
    return f"https://www.google.com/maps/search/?api=1&query={query}"

st.set_page_config(page_title="Leightonfield Sorting", layout="wide")

# --- Sidebar ---
st.sidebar.header("Search Settings")
option = st.sidebar.selectbox("Search by:", ["Street Address", "Beat Number", "Suburb"])
debug_mode = st.sidebar.checkbox("Show AI View (Debug)", value=False)

# --- OCR Section ---
scanned_street = None
scanned_no = None

if option == "Street Address" and OCR_AVAILABLE:
    with st.expander("📸 Scan Address Label", expanded=True):
        img_file = st.camera_input("Take photo")
        
        if img_file:
            img = Image.open(img_file)
            # Add Cropper
            st.write("### 1. Crop to the Address")
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
            
            if st.button("🚀 Process Crop"):
                img_np = np.array(cropped_img)
                
                with st.spinner("Analyzing..."):
                    # Enhance Image
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    
                    if debug_mode: st.image(enhanced, caption="AI View")

                    # OCR
                    results = reader.readtext(enhanced, paragraph=False, contrast_ths=0.05, adjust_contrast=0.8)
                    detected_strings = [normalize_text(res[1]) for res in results]
                    full_blob = " ".join(detected_strings)
                    
                    # Fuzzy Match
                    best_match, score = process.extractOne(full_blob, street_list, scorer=fuzz.partial_ratio)
                    
                    if score > 60:
                        scanned_street = best_match
                        
                        # Find Number (Unit-Aware)
                        street_box_idx = next((i for i, t in enumerate(detected_strings) if fuzz.partial_ratio(scanned_street, t) > 75), -1)
                        if street_box_idx != -1:
                            context = " ".join(detected_strings[max(0, street_box_idx-1):street_box_idx+1]).replace('/', ' ')
                            nums = re.findall(r'\b(\d+)\b', context)
                            potential = [n for n in nums if not (len(n) == 4 and n.startswith('2'))]
                            scanned_no = potential[-1] if potential else (nums[-1] if nums else None)

                        st.success(f"✅ Found: {scanned_no if scanned_no else ''} {scanned_street} ({score}%)")
                    else:
                        st.error("Could not match street name. Try a tighter crop.")

# --- Search Logic ---
results_df = pd.DataFrame()
searched_no = None

if option == "Street Address":
    col1, col2 = st.columns([3, 1])
    with col1:
        st_name = st.selectbox("Street Name", options=street_list, index=street_list.index(scanned_street) if scanned_street in street_list else None)
    with col2:
        st_no_str = st.text_input("Number", value=scanned_no if scanned_no else "")
        if st_no_str.isdigit(): searched_no = int(st_no_str)
   
    if st_name:
        if searched_no is not None:
            parity = 2 if searched_no % 2 == 0 else 1
            results_df = df[(df['StreetName'] == st_name) & (df['EvenOdd'] == parity) & (df['StreetNoMin'] <= searched_no) & (df['StreetNoMax'] >= searched_no)].copy()
        else:
            results_df = df[df['StreetName'] == st_name].copy()

elif option == "Beat Number":
    beat_val = st.sidebar.number_input("Enter Beat Number", min_value=1, step=1)
    results_df = df[df['BeatNo'] == beat_val].copy()

elif option == "Suburb":
    sub_val = st.selectbox("Select Suburb", sorted(df['Suburb'].unique()), index=None)
    results_df = df[df['Suburb'] == sub_val].copy()

# --- Display Results ---
st.divider()
if not results_df.empty:
    results_df['Map Link'] = results_df.apply(lambda row: make_map_link(row, searched_no), axis=1)
    st.dataframe(
        results_df[['Map Link', 'BeatNo', 'Postcode', 'Suburb', 'StreetName', 'StreetNoMin', 'StreetNoMax']],
        column_config={"Map Link": st.column_config.LinkColumn("Maps", display_text="📍 View")},
        use_container_width=True, hide_index=True
    )
    
 

 
































