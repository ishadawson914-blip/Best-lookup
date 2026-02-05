import streamlit as st
import pandas as pd
import urllib.parse
import numpy as np
import re
from PIL import Image
from thefuzz import process, fuzz

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
        return easyocr.Reader(['en'])
    return None

# Load the data
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
    return f"http://maps.google.com/?q={query}"

st.set_page_config(page_title="Leightonfield Sorting", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar Search Settings ---
st.sidebar.header("Search Settings")
option = st.sidebar.selectbox("Search by:", ["Street Address", "Beat Number", "Suburb"])

# --- OCR Camera Section ---
scanned_street = None
scanned_no = None

if option == "Street Address" and OCR_AVAILABLE:
    with st.expander("📸 Scan Address from Photo (Label)", expanded=True):
        img_file = st.camera_input("Take a photo of the address label")
        
        if img_file:
            img = Image.open(img_file)
            img_np = np.array(img)
            
            with st.spinner("Analyzing image..."):
                results_ocr = reader.readtext(img_np)
                detected_strings = [res[1].upper().strip() for res in results_ocr]
                full_text_blob = " ".join(detected_strings)
                
                # 1. Fuzzy Street Match (Less strict threshold of 60)
                best_match, score = process.extractOne(full_text_blob, street_list, scorer=fuzz.partial_ratio)
                
                if score > 60:
                    scanned_street = best_match
                    
                    # 2. Find the index of the box containing the street name
                    street_box_idx = -1
                    for i, text in enumerate(detected_strings):
                        if fuzz.partial_ratio(scanned_street, text) > 70:
                            street_box_idx = i
                            break
                    
                    # 3. Targeted Number Search (ONLY BEFORE the street name)
                    if street_box_idx != -1:
                        # Look at the box containing the street and the 2 boxes before it
                        start_search = max(0, street_box_idx - 2)
                        # We only look up to the street box itself
                        context_text = " ".join(detected_strings[start_search : street_box_idx + 1])
                        
                        # Find all numbers
                        found_numbers = re.findall(r'\b(\d+)\b', context_text)
                        
                        if found_numbers:
                            # Filter out common NSW postcodes (usually 2000-2999) 
                            # if they are the only number, we'll take them, 
                            # but we prefer the FIRST number in the sequence (House Number)
                            potential_numbers = [n for n in found_numbers if not (len(n) == 4 and n.startswith('2'))]
                            
                            if potential_numbers:
                                scanned_no = potential_numbers[0] # Take the FIRST one found before street
                            else:
                                scanned_no = found_numbers[0]

                    # Feedback
                    if score > 85:
                        st.success(f"✅ Found: **{scanned_no if scanned_no else ''} {scanned_street}** ({score}%)")
                    else:
                        st.info(f"🤔 Best Guess: **{scanned_no if scanned_no else ''} {scanned_street}** ({score}%)")
                else:
                    st.error("Could not identify street. Try a clearer photo.")

# --- Search Logic ---
results = pd.DataFrame()
searched_no = None

if option == "Street Address":
    col1, col2 = st.columns([3, 1])
    with col1:
        st_name = st.selectbox(
            "Street Name",
            options=street_list,
            index=street_list.index(scanned_street) if scanned_street in street_list else None,
            placeholder="Select or scan"
        )
    with col2:
        st_no_str = st.text_input("Number", value=scanned_no if scanned_no else "")
        if st_no_str.isdigit():
            searched_no = int(st_no_str)
   
    if st_name:
        if searched_no is not None:
            parity = 2 if searched_no % 2 == 0 else 1
            mask = (
                (df['StreetName'] == st_name) &
                (df['EvenOdd'] == parity) &
                (df['StreetNoMin'] <= searched_no) &
                (df['StreetNoMax'] >= searched_no)
            )
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

# --- Results Table ---
st.divider()
if not results.empty:
    results = results.sort_values(by=['Suburb', 'StreetName', 'StreetNoMin'])
    results['Map Link'] = results.apply(lambda row: make_map_link(row, searched_no), axis=1)
    
    display_cols = ['Map Link', 'BeatNo', 'Postcode', 'Suburb', 'StreetName', 'StreetNoMin', 'StreetNoMax', 'TeamNo']
    st.dataframe(
        results[display_cols],
        column_config={
            "Map Link": st.column_config.LinkColumn("Maps", display_text="📍 View"),
            "StreetNoMin": "From (Min No)", "StreetNoMax": "To (Max No)"
        },
        use_container_width=True, hide_index=True
    )
    
 

 
























