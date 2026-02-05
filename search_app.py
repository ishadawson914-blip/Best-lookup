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
    # Ensure SortCart.csv is in the same directory
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

# --- OCR Camera Section ---
scanned_street = None
scanned_no = None
confidence_score = 0

if OCR_AVAILABLE:
    with st.expander("📸 Scan Address from Photo (Label)", expanded=True):
        img_file = st.camera_input("Take a photo of the address label")
        
        if img_file:
            img = Image.open(img_file)
            img_np = np.array(img)
            
            with st.spinner("Analyzing image..."):
                results_ocr = reader.readtext(img_np)
                
                # 1. Extract and clean text strings
                detected_strings = [res[1].upper().strip() for res in results_ocr]
                full_text_blob = " ".join(detected_strings)
                
                # 2. Fuzzy Street Match
                # Uses thefuzz to find the closest match from your master street_list
                best_match, score = process.extractOne(full_text_blob, street_list, scorer=fuzz.partial_ratio)
                
                if score > 70:  # If we have a decent match
                    scanned_street = best_match
                    confidence_score = score
                    
                    # 3. Smart Number Extraction based on proximity
                    # Find which OCR box likely contained the street name
                    street_box_idx = -1
                    for i, text in enumerate(detected_strings):
                        if fuzz.partial_ratio(scanned_street, text) > 80:
                            street_box_idx = i
                            break
                    
                    # Look for numbers in the box before or inside the street name box
                    # This avoids picking up postcodes or phone numbers elsewhere on the label
                    context_text = ""
                    if street_box_idx != -1:
                        start = max(0, street_box_idx - 1)
                        end = street_box_idx + 1
                        context_text = " ".join(detected_strings[start:end])
                    
                    # Find sequences of digits
                    found_numbers = re.findall(r'\b(\d+)\b', context_text)
                    if found_numbers:
                        # Usually the house number is the last digit sequence before/at the street name
                        scanned_no = found_numbers[-1]

                    # 4. Display Confidence Feedback
                    if score > 90:
                        st.success(f"✅ Found: **{scanned_no if scanned_no else ''} {scanned_street}** (Confidence: {score}%)")
                    else:
                        st.warning(f"⚠️ Likely: **{scanned_no if scanned_no else ''} {scanned_street}** ({score}% match). Please check below.")
                else:
                    st.error("Could not clearly identify a street name. Try a clearer photo.")

else:
    st.warning("OCR libraries not detected. Please check requirements.txt")

# --- Search Settings ---
st.sidebar.header("Search Settings")
option = st.sidebar.selectbox("Search by:", ["Street Address", "Beat Number", "Suburb"])

results = pd.DataFrame()
searched_no = None

if option == "Street Address":
    col1, col2 = st.columns([3, 1])
    with col1:
        # We pre-fill the selectbox if the scan was successful
        st_name = st.selectbox(
            "Street Name",
            options=street_list,
            index=street_list.index(scanned_street) if scanned_street in street_list else None,
            placeholder="Select or scan a street"
        )
    with col2:
        # Pre-fill the number from the scan
        default_no = scanned_no if scanned_no else ""
        st_no_str = st.text_input("Number", value=default_no)
        if st_no_str.isdigit():
            searched_no = int(st_no_str)
   
    if st_name:
        if searched_no is not None:
            # Check for Even/Odd parity and range
            parity = 2 if searched_no % 2 == 0 else 1
            mask = (
                (df['StreetName'] == st_name) &
                (df['EvenOdd'] == parity) &
                (df['StreetNoMin'] <= searched_no) &
                (df['StreetNoMax'] >= searched_no)
            )
            results = df[mask].copy()
        else:
            # If no number entered, show all segments for that street
            results = df[df['StreetName'] == st_name].copy()

elif option == "Beat Number":
    beat_val = st.sidebar.number_input("Enter Beat Number", min_value=1, step=1)
    results = df[df['BeatNo'] == beat_val].copy()

elif option == "Suburb":
    suburb_list = sorted(df['Suburb'].unique())
    sub_val = st.selectbox("Select Suburb", suburb_list, index=None, placeholder="Choose a suburb")
    results = df[df['Suburb'] == sub_val].copy()

# --- Display Results ---
st.divider()

if not results.empty:
    results = results.sort_values(by=['Suburb', 'StreetName', 'StreetNoMin'])
    results['Map Link'] = results.apply(lambda row: make_map_link(row, searched_no), axis=1)
   
    st.success(f"Found {len(results)} matching sorting record(s)")
    
    display_cols = ['Suburb', 'StreetName', 'StreetNoMin', 'StreetNoMax', 'BeatNo', 'TeamNo', 'Postcode', 'Map Link']
    
    st.dataframe(
        results[display_cols],
        column_config={
            "Map Link": st.column_config.LinkColumn("Maps", display_text="📍 View"),
            "BeatNo": "Beat", 
            "TeamNo": "Team",
            "StreetNoMin": "Min", 
            "StreetNoMax": "Max"
        },
        use_container_width=True,
        hide_index=True
    )
   
    csv = results[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export Results", data=csv, file_name='search_results.csv', mime='text/csv')

elif (option == "Street Address" and st_name):
    st.error(f"No entry found for {st_no_str} {st_name}. Please verify the number is within range.")
    
 

 




















