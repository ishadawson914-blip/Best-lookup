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
        # gpu=False is required for Streamlit Cloud Free Tier
        return easyocr.Reader(['en'], gpu=False)
    return None

# --- Data Loading & Normalization ---
@st.cache_data
def load_data():
    df = pd.read_csv('SortCart.csv')
    unique_streets = sorted(df['StreetName'].unique())
    return df, unique_streets

# Mapping common OCR misreads or abbreviations
SUFFIX_MAP = {
    'ST': 'STREET', 'RD': 'ROAD', 'AVE': 'AVENUE', 'DR': 'DRIVE',
    'PL': 'PLACE', 'CT': 'COURT', 'HWY': 'HIGHWAY', 'CR': 'CRESCENT',
    'CRES': 'CRESCENT', 'CL': 'CLOSE', 'GR': 'GROVE', 'PDE': 'PARADE'
}

def normalize_text(text):
    if not text: return ""
    words = text.upper().split()
    normalized = [SUFFIX_MAP.get(w, w) for w in words]
    return " ".join(normalized)

df, street_list = load_data()
reader = load_ocr()

def make_map_link(row, specific_no=None):
    num = specific_no if specific_no else row['StreetNoMin']
    address = f"{num} {row['StreetName']}, {row['Suburb']}, NSW {row['Postcode']}, Australia"
    query = urllib.parse.quote(address)
    return f"https://www.google.com/maps/search/?api=1&query={query}"

# --- Page Layout ---
st.set_page_config(page_title="Leightonfield Sorting", layout="wide")

st.title("📦 Leightonfield Address Sorter")

# --- Sidebar ---
st.sidebar.header("Search Settings")
option = st.sidebar.selectbox("Search by:", ["Street Address", "Beat Number", "Suburb"])
debug_mode = st.sidebar.checkbox("Show AI View (Debug)", value=False)

scanned_street = None
scanned_no = None

# --- OCR Camera & Cropping Section ---
if option == "Street Address" and OCR_AVAILABLE:
    with st.expander("📸 Scan Address Label", expanded=True):
        img_file = st.camera_input("Take a photo of the label")
        
        if img_file:
            img = Image.open(img_file)
            
            st.warning("Manual Step: Crop the box to show ONLY the destination address.")
            # aspect_ratio=None allows free-form cropping
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
            
            if st.button("🚀 Process Scanned Address"):
                img_np = np.array(cropped_img)
                
                with st.spinner("Enhancing image and reading text..."):
                    # 1. Pre-processing for low-light/glare
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    
                    if debug_mode:
                        st.image(enhanced, caption="AI Enhanced View", use_container_width=True)

                    # 2. OCR Detection
                    results_ocr = reader.readtext(
                        enhanced, 
                        paragraph=False, 
                        contrast_ths=0.05, 
                        adjust_contrast=0.8
                    )
                    
                    detected_strings = [normalize_text(res[1]) for res in results_ocr]
                    full_text_blob = " ".join(detected_strings)
                    
                    # 3. Fuzzy Street Matching
                    best_match, score = process.extractOne(full_text_blob, street_list, scorer=fuzz.partial_ratio)
                    
                    if score > 60:
                        scanned_street = best_match
                        
                        # Find the Street Number near the detected street name
                        street_box_idx = -1
                        for i, text in enumerate(detected_strings):
                            if fuzz.partial_ratio(scanned_street, text) > 75:
                                street_box_idx = i
                                break
                        
                        if street_box_idx != -1:
                            # Look at current and previous 2 blocks for a number
                            context = " ".join(detected_strings[max(0, street_box_idx-2) : street_box_idx+1])
                            clean_context = context.replace('/', ' ')
                            found_numbers = re.findall(r'\b(\d+)\b', clean_context)
                            
                            if found_numbers:
                                # Filter out likely postcodes (2000-2999)
                                potential = [n for n in found_numbers if not (len(n) == 4 and n.startswith('2'))]
                                scanned_no = potential[-1] if potential else found_numbers[-1]

                        st.success(f"✅ Found Match: **{scanned_no if scanned_no else ''} {scanned_street}** ({score}% confidence)")
                    else:
                        st.error("Could not recognize street. Please crop tighter or try a clearer photo.")

# --- Search Logic & Results ---
results_df = pd.DataFrame()
searched_no = None

if option == "Street Address":
    col1, col2 = st.columns([3, 1])
    with col1:
        st_name = st.selectbox(
            "Street Name", 
            options=street_list, 
            index=street_list.index(scanned_street) if scanned_street in street_list else None
        )
    with col2:
        st_no_str = st.text_input("Number", value=scanned_no if scanned_no else "")
        if st_no_str.isdigit():
            searched_no = int(st_no_str)
   
    if st_name:
        if searched_no is not None:
            parity = 2 if searched_no % 2 == 0 else 1
            mask = (df['StreetName'] == st_name) & (df['EvenOdd'] == parity) & \
                   (df['StreetNoMin'] <= searched_no) & (df['StreetNoMax'] >= searched_no)
            results_df = df[mask].copy()
        else:
            results_df = df[df['StreetName'] == st_name].copy()

elif option == "Beat Number":
    beat_val = st.sidebar.number_input("Enter Beat Number", min_value=1, step=1)
    results_df = df[df['BeatNo'] == beat_val].copy()

elif option == "Suburb":
    sub_val = st.selectbox("Select Suburb", sorted(df['Suburb'].unique()), index=None)
    results_df = df[df['Suburb'] == sub_val].copy()

# --- Big Result Display ---
st.divider()

if not results_df.empty:
    # Highlight the primary result if it's a specific address match
    if searched_no:
        primary_beat = results_df.iloc[0]['BeatNo']
        st.markdown(f"""
            <div style="background-color:#007BFF;padding:20px;border-radius:10px;text-align:center;">
                <h1 style="color:white;margin:0;">BEAT: {primary_beat}</h1>
                <p style="color:white;margin:0;">Team: {results_df.iloc[0]['TeamNo']}</p>
            </div>
            """, unsafe_content_allowed=True)
        st.write("")

    results_df['Map Link'] = results_df.apply(lambda row: make_map_link(row, searched_no), axis=1)
    
    st.dataframe(
        results_df[['Map Link', 'BeatNo', 'Postcode', 'Suburb', 'StreetName', 'StreetNoMin', 'StreetNoMax', 'TeamNo']],
        column_config={"Map Link": st.column_config.LinkColumn("Maps", display_text="📍 View")},
        use_container_width=True, hide_index=True
    )
elif scanned_street or searched_no:
    st.warning("No sorting record found for this specific address.")
    
 

 

































