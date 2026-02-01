import streamlit as st
import pandas as pd
import urllib.parse
import numpy as np
import re
from PIL import Image

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
        # This downloads the models on the first run
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
    return f"https://www.google.com/maps/search/?api=1&query={query}"

st.set_page_config(page_title="Leightonfield", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a cleaner mobile look
st.markdown('''
    <style>
    .stSelectbox div[data-baseweb="select"] { background-color: white; }
    </style>
    ''', unsafe_allow_html=True)

st.title("📍 Sorting App with Photo Scan")

# --- OCR Camera Section ---
scanned_street = None
scanned_no = None

if OCR_AVAILABLE:
    with st.expander("📸 Scan Address from Photo (Label)"):
        img_file = st.camera_input("Take a photo of the address")
        if img_file:
            img = Image.open(img_file)
            img_np = np.array(img)
            with st.spinner("Analyzing image..."):
                results_ocr = reader.readtext(img_np)
                # Sort OCR results by their vertical position then horizontal
                results_ocr.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
                full_text = " ".join([res[1].upper() for res in results_ocr])
                st.info(f"Detected Text: {full_text}")

                # Improved extraction logic
                # 1. Find the street name first
                found_street = None
                for street in street_list:
                    if street in full_text:
                        # Find the longest matching street name to be more specific
                        if found_street is None or len(street) > len(found_street):
                            found_street = street
                
                if found_street:
                    scanned_street = found_street
                    # 2. Look for a number near the street name
                    # Look for digits in the 25 characters preceding the street name
                    pattern = re.compile(r'(\d+)')
                    idx = full_text.find(found_street)
                    
                    # Search in a window before the street name
                    prefix = full_text[max(0, idx-25):idx]
                    numbers = pattern.findall(prefix)
                    if numbers:
                        # Take the last number before the street name (most likely house number)
                        scanned_no = numbers[-1]
                    else:
                        # Fallback: check shortly after the street name
                        suffix = full_text[idx + len(found_street): idx + len(found_street) + 15]
                        numbers_suffix = pattern.findall(suffix)
                        if numbers_suffix:
                            scanned_no = numbers_suffix[0]
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
        st_name = st.selectbox(
            "Street Name",
            options=street_list,
            index=street_list.index(scanned_street) if scanned_street in street_list else None,
            placeholder="Select or scan a street"
        )
    with col2:
        default_no = scanned_no if scanned_no else ""
        st_no_str = st.text_input("Number (optional)", value=default_no)
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
    beat_val = st.sidebar.number_input("Enter Beat Number", min_value=1, value=1011)
    results = df[df['BeatNo'] == beat_val].copy()

elif option == "Suburb":
    suburb_list = sorted(df['Suburb'].unique())
    sub_val = st.selectbox("Select Suburb", suburb_list, index=None, placeholder="Choose a suburb")
    results = df[df['Suburb'] == sub_val].copy()

# --- Display Results ---
if not results.empty:
    # Hierarchical Sorting
    results = results.sort_values(by=['Suburb', 'StreetName', 'StreetNoMin'])
    results['Map Link'] = results.apply(lambda row: make_map_link(row, searched_no), axis=1)
   
    st.success(f"Found {len(results)} record(s)")
    
    # Column selection: Suburb is first
    display_cols = ['Suburb', 'StreetName', 'StreetNoMin', 'StreetNoMax', 'BeatNo', 'TeamNo', 'Postcode', 'Map Link']
    display_results = results[display_cols]
   
    st.dataframe(
        display_results,
        column_config={
            "Map Link": st.column_config.LinkColumn("Maps", display_text="📍 View"),
            "BeatNo": "Beat", "TeamNo": "Team",
            "StreetNoMin": "Min No", "StreetNoMax": "Max No"
        },
        use_container_width=True,
        hide_index=True
    )
   
    csv = display_results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export to CSV", data=csv, file_name='search_results.csv', mime='text/csv')

elif (option == "Street Address" and st_name):
    st.warning("No entry found. Please check the street number.")
    
 

 



















