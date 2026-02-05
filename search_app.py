import streamlit as st
import pandas as pd
import urllib.parse
import numpy as np
import re
from PIL import Image
from thefuzz import process, fuzz

# --- OCR Setup ---
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

@st.cache_resource
def load_ocr():
    if OCR_AVAILABLE:
        return easyocr.Reader(['en'])
    return None

@st.cache_data
def load_data():
    df = pd.read_csv('SortCart.csv')
    unique_streets = sorted(df['StreetName'].unique())
    return df, unique_streets

df, street_list = load_data()
reader = load_ocr()

def make_map_link(row, specific_no=None):
    num = specific_no if specific_no else row['StreetNoMin']
    address = f"{num} {row['StreetName']}, {row['Suburb']}, NSW {row['Postcode']}, Australia"
    query = urllib.parse.quote(address)
    # Using '0' or '1' prefix depending on your preferred Google Maps behavior
    return f"http://maps.google.com/?q={query}"

st.set_page_config(page_title="Leightonfield Sorting", layout="wide")

# --- Sidebar Search Settings ---
st.sidebar.header("Search Settings")
option = st.sidebar.selectbox("Search by:", ["Street Address", "Beat Number", "Suburb"])

scanned_street = None
scanned_no = None

# --- OCR Section (Both Camera and File Upload) ---
if option == "Street Address" and OCR_AVAILABLE:
    with st.expander("📸 Scan Address Label", expanded=True):
        col_cam, col_file = st.columns(2)
        
        with col_cam:
            cam_file = st.camera_input("Quick Scan (Selfie/Front)")
        with col_file:
            up_file = st.file_uploader("Upload / Back Camera", type=['jpg', 'jpeg', 'png'])
        
        # Use whichever input has a file
        img_file = up_file if up_file else cam_file
        
        if img_file:
            img = Image.open(img_file)
            img_np = np.array(img)
            
            with st.spinner("Analyzing address..."):
                results_ocr = reader.readtext(img_np)
                detected_strings = [res[1].upper().strip() for res in results_ocr]
                full_text_blob = " ".join(detected_strings)
                
                # 1. Fuzzy Street Match (Threshold 60 for less strict matching)
                best_match, score = process.extractOne(full_text_blob, street_list, scorer=fuzz.partial_ratio)
                
                if score > 60:
                    scanned_street = best_match
                    
                    # 2. Locate street name in OCR results
                    street_box_idx = -1
                    for i, text in enumerate(detected_strings):
                        if fuzz.partial_ratio(scanned_street, text) > 75:
                            street_box_idx = i
                            break
                    
                    # 3. Unit-Aware Street Number Logic
                    if street_box_idx != -1:
                        # Search context (2 boxes before the street name)
                        start_search = max(0, street_box_idx - 2)
                        context_text = " ".join(detected_strings[start_search : street_box_idx + 1])
                        
                        # Clean common unit prefixes and slashes (e.g., 1603/28)
                        clean_context = re.sub(r'[/|UNIT|APT|SUITE|U]', ' ', context_text)
                        found_numbers = re.findall(r'\b(\d+)\b', clean_context)
                        
                        if found_numbers:
                            # Filter out NSW postcodes (2000-2999)
                            potential_numbers = [n for n in found_numbers if not (len(n) == 4 and n.startswith('2'))]
                            
                            if potential_numbers:
                                # Grab the LAST number before the street (ignores the unit number)
                                scanned_no = potential_numbers[-1]
                            else:
                                scanned_no = found_numbers[-1]

                    if score > 85:
                        st.success(f"✅ Found: **{scanned_no if scanned_no else ''} {scanned_street}** ({score}%)")
                    else:
                        st.info(f"🤔 Best Guess: **{scanned_no if scanned_no else ''} {scanned_street}** ({score}%)")

# --- Search UI ---
results = pd.DataFrame()
searched_no = None

if option == "Street Address":
    col1, col2 = st.columns([3, 1])
    with col1:
        st_name = st.selectbox(
            "Street Name", options=street_list,
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
            results = df[mask].copy()
        else:
            results = df[df['StreetName'] == st_name].copy()

elif option == "Beat Number":
    beat_val = st.sidebar.number_input("Enter Beat Number", min_value=1, step=1)
    results = df[df['BeatNo'] == beat_val].copy()

elif option == "Suburb":
    suburb_list = sorted(df['Suburb'].unique())
    sub_val = st.selectbox("Select Suburb", suburb_list, index=None)
    results = df[df['Suburb'] == sub_val].copy()

# --- Display Results ---
st.divider()
if not results.empty:
    results = results.sort_values(by=['Suburb', 'StreetName', 'StreetNoMin'])
    results['Map Link'] = results.apply(lambda row: make_map_link(row, searched_no), axis=1)
    
    # Column Order: Maps, Beat, Postcode, Suburb, Street Name, From, To, Team
    display_cols = ['Map Link', 'BeatNo', 'Postcode', 'Suburb', 'StreetName', 'StreetNoMin', 'StreetNoMax', 'TeamNo']
    st.dataframe(
        results[display_cols],
        column_config={
            "Map Link": st.column_config.LinkColumn("Maps", display_text="📍 View"),
            "StreetNoMin": "From", "StreetNoMax": "To", "BeatNo": "Beat", "TeamNo": "Team"
        },
        use_container_width=True, hide_index=True
    )
    
 

 





























