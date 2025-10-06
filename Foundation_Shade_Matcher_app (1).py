import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import to_rgb, hex2color
from skimage import color
import re

# -----------------------------
# Page config & small helpers
# -----------------------------
st.set_page_config(page_title="💄 Foundation Recommender (Phase 3)", layout="wide",initial_sidebar_state="expanded"
)

def safe_hex_to_rgb(hex_code):
    try:
        return np.array(hex2color(hex_code))
    except Exception:
        # fallback to neutral mid-gray
        return np.array([0.5, 0.5, 0.5])

def rgb_to_lab(rgb_0_1):
    # rgb_0_1 expected shape (3,) or (1,3) with values in [0,1]
    arr = np.array(rgb_0_1).reshape(1,1,3)
    lab = color.rgb2lab(arr).reshape(3,)
    return lab  # L,a,b

def text_contains(tokens, text):
    text = str(text).lower()
    return any(t in text for t in tokens)

# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "merged.csv"
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset '{DATA_PATH}' not found. Upload merged.csv to the app folder.")
    st.stop()

# Ensure expected columns exist
for c in ['hex','brand','product','name','description','categories','url']:
    if c not in df.columns:
        df[c] = ""

# Normalize hex lower-case
df['hex'] = df['hex'].astype(str).str.strip().str.lower()

# -----------------------------
# Build color features (Lab)
# -----------------------------
# If Lab columns exist (L_s, A_s, B_s) use them; otherwise compute from hex
if all(col in df.columns for col in ['L_s','A_s','B_s']):
    lab_matrix = df[['L_s','A_s','B_s']].values
else:
    # compute Lab from hex
    lab_list = []
    for h in df['hex'].values:
        rgb = safe_hex_to_rgb(h)  # 0-1
        lab = rgb_to_lab(rgb)
        lab_list.append(lab)
    lab_matrix = np.vstack(lab_list)

# Build k-NN (Lab space)
knn = NearestNeighbors(n_neighbors=12, metric='euclidean', algorithm='auto')
knn.fit(lab_matrix)

# -----------------------------
# UI: header + instructions
# -----------------------------
st.markdown("""
    <div style="text-align:center">
        <h1 style="color:#B58FA5">💄 Foundation Shade Assistant</h1>
        <p style="color:#666">Personalized shade recommendations using color science (CIELAB) + simple beauty filters.</p>
    </div>
""", unsafe_allow_html=True)

# Two input modes: RGB pick or Brand+Shade select
mode = st.radio("How would you like to provide a sample tone?", ("🎨 Use RGB Sliders", "🏷 Select Brand & Shade"))

# Sidebar filters (undertone, finish, brands)
st.sidebar.header("Filters & Preferences")
undertone_pref = st.sidebar.selectbox("Your undertone (optional)", ["Any", "Warm", "Neutral", "Cool"])
finish_pref = st.sidebar.selectbox("Preferred finish (optional)", ["Any", "Matte", "Natural", "Dewy"])
brand_pref_list = sorted(df['brand'].dropna().unique())
brand_pref = st.sidebar.multiselect("Prefer these brands (optional)", options=brand_pref_list, default=[])

# Small helper to score matches for finish/undertone from description/categories
def label_from_text(text):
    t = str(text).lower()
    tone = []
    if any(k in t for k in ['warm','gold','golden','peach','yellow','yellowish','warm tone','warm undertone','warm-toned']):
        tone.append('Warm')
    if any(k in t for k in ['cool','pink','rosy','blue','cool undertone','cool-toned']):
        tone.append('Cool')
    if any(k in t for k in ['neutral','neutral undertone','universal']):
        tone.append('Neutral')
    # finish detection
    finish = None
    if any(k in t for k in ['matte','matte finish','velvet']):
        finish = 'Matte'
    if any(k in t for k in ['dewy','glow','radiant','luminous','hydrating']):
        finish = 'Dewy'
    if any(k in t for k in ['natural','sheer','natural finish','semi-matte']):
        finish = 'Natural'
    return tone, finish

# Precompute textual labels for performance
text_labels = []
for i, row in df.iterrows():
    txt = " ".join([str(row.get('description','')) , str(row.get('categories','')), str(row.get('product','')), str(row.get('name',''))]).lower()
    tone_tokens = []
    if any(k in txt for k in ['warm','gold','peach','yellow']):
        tone_tokens.append('Warm')
    if any(k in txt for k in ['cool','pink','rosy','blue']):
        tone_tokens.append('Cool')
    if any(k in txt for k in ['neutral','universal']):
        tone_tokens.append('Neutral')
    # finish
    finish_label = None
    if any(k in txt for k in ['matte']):
        finish_label = 'Matte'
    elif any(k in txt for k in ['dewy','glow','luminous']):
        finish_label = 'Dewy'
    elif any(k in txt for k in ['natural','sheer']):
        finish_label = 'Natural'
    text_labels.append({'tones': tone_tokens, 'finish': finish_label})
# attach to DataFrame (not necessary but convenient)
df = df.reset_index(drop=True)
df['_tones'] = [x['tones'] for x in text_labels]
df['_finish'] = [x['finish'] for x in text_labels]

# -----------------------------
# Recommendation core
# -----------------------------
def get_recommendations_from_lab(lab_input, top_n=8, brand_filter=None, undertone_filter="Any", finish_filter="Any"):
    distances, indices = knn.kneighbors(lab_input.reshape(1,-1), n_neighbors=top_n+4)  # get a few extras to filter
    cand_idx = indices[0]
    cand_scores = distances[0]
    # Compose list and filter by optional criteria
    results = []
    for idx, dist in zip(cand_idx, cand_scores):
        row = df.iloc[idx].to_dict()
        # Filter by brand preference if requested
        if brand_filter and len(brand_filter) > 0 and row['brand'] not in brand_filter:
            continue
        # Filter by undertone if specified (check text labels)
        if undertone_filter != "Any":
            tones = df.loc[idx, '_tones']
            if undertone_filter not in tones:
                # allow near-miss: if neutral requested, allow anything
                if undertone_filter != 'Neutral':
                    continue
        # Filter by finish
        if finish_filter != "Any":
            f = df.loc[idx,'_finish']
            if f is None or f != finish_filter:
                continue
        results.append((idx, dist))
        if len(results) >= top_n:
            break
    # Build DataFrame of results
    if not results:
        return pd.DataFrame()
    idxs, dists = zip(*results)
    res = df.iloc[list(idxs)].copy()
    res['distance'] = list(dists)
    return res

# -----------------------------
# UI: collect input & run
# -----------------------------
selected_lab = None
selected_color_hex = None
selected_source = None

if mode.startswith("🎨"):
    st.markdown("### Pick a sample skin-tone (RGB sliders)")
    r = st.slider("Red", 0, 255, 210)
    g = st.slider("Green", 0, 255, 160)
    b = st.slider("Blue", 0, 255, 130)
    rgb01 = np.array([r/255.0, g/255.0, b/255.0])
    selected_lab = rgb_to_lab(rgb01)
    # create display box
    st.markdown(f"<div style='width:120px; height:80px; border-radius:8px; background-color: rgb({r},{g},{b}); margin:auto; border:1px solid #ccc'></div>", unsafe_allow_html=True)
    selected_color_hex = '#%02x%02x%02x' % (r, g, b)
    selected_source = 'rgb'
else:
    # Brand + Shade selection
    st.markdown("### Choose a brand and select one of its shades")
    brand_list = sorted(df['brand'].dropna().unique())
    brand_choice = st.selectbox("Brand", brand_list)
    shades_for_brand = df.loc[df['brand']==brand_choice, 'name'].dropna().unique()
    if len(shades_for_brand)==0:
        st.warning("No shades available for this brand in the dataset.")
    shade_choice = st.selectbox("Shade", sorted(shades_for_brand))
    # find first matching shade row
    row = df.loc[(df['brand']==brand_choice) & (df['name']==shade_choice)]
    if row.empty:
        st.error("Selected shade not found.")
    else:
        row0 = row.iloc[0]
        selected_color_hex = row0['hex']
        rgb01 = safe_hex_to_rgb(selected_color_hex)
        selected_lab = rgb_to_lab(rgb01)
        selected_source = 'brand_shade'
        # show preview
        st.markdown(f"<div style='width:120px; height:80px; border-radius:8px; background-color: {selected_color_hex}; margin:auto; border:1px solid #ccc'></div>", unsafe_allow_html=True)

# Run recommendation when button clicked
run_button = st.button("✨ Get My Recommendations")

if run_button:
    if selected_lab is None:
        st.error("No input provided.")
    else:
        with st.spinner("Matching shades — one sec..."):
            time.sleep(0.8)
            recs = get_recommendations_from_lab(selected_lab, top_n=6, brand_filter=brand_pref, undertone_filter=undertone_pref, finish_filter=finish_pref)
        if recs.empty:
            st.warning("No matches found with current filters — try relaxing filters or broadening brand preferences.")
        else:
            # Provide high-level personalized messages
            tone_msg = {
                "Warm": "Golden/peachy undertones — warm shades usually enhance warmth and glow.",
                "Cool": "Pink/rosy undertones — cool shades often neutralize redness and look balanced.",
                "Neutral": "Neutral undertones — you can typically wear both warm and cool shades."
            }
            finish_msg = {
                "Matte": "You prefer matte finish — we prioritize longwear, low-shine formulas.",
                "Natural": "You prefer a natural finish — we favor blendable, medium-coverage formulas.",
                "Dewy": "You prefer a dewy finish — we favor hydrating, luminous formulas."
            }
            # Show summary
            st.markdown("## Your Personalized Recommendations")
            if undertone_pref != "Any":
                st.info(tone_msg.get(undertone_pref, ""))
            if finish_pref != "Any":
                st.info(finish_msg.get(finish_pref, ""))

            # show selected shade card
            st.markdown("### Selected / Input Tone")
            sel_html = f"""
                <div style="display:flex;align-items:center;gap:15px;padding:12px;border-radius:12px;background:#fff;border:1px solid #eee">
                    <div style="width:90px;height:70px;border-radius:8px;background:{selected_color_hex};border:1px solid #ccc"></div>
                    <div>
                        <div style="font-weight:700">{'Custom RGB sample' if selected_source=='rgb' else f'{brand_choice} — {shade_choice}'}</div>
                        <div style="color:#666;margin-top:4px">HEX: {selected_color_hex}</div>
                    </div>
                </div>
            """
            st.markdown(sel_html, unsafe_allow_html=True)

            # Display recommendations in 3-column grid
            st.markdown("### Matches")
            cols = st.columns(3)
            for i, (_, rrow) in enumerate(recs.iterrows()):
                col = cols[i % 3]
                with col:
                    # determine accent color based on shade lightness
                    try:
                        rgb01_r = safe_hex_to_rgb(rrow['hex'])
                        lightness = 0.2126 * rgb01_r[0] + 0.7152 * rgb01_r[1] + 0.0722 * rgb01_r[2]
                        if lightness > 0.7:
                            accent = "#EFB9B9"
                        elif lightness > 0.4:
                            accent = "#D7A4C3"
                        else:
                            accent = "#B58FA5"
                    except:
                        accent = "#D7A4C3"

                    card_html = f"""
                        <div style="border-radius:12px;padding:12px;background:#fff;border:1px solid #eee;margin-bottom:12px;">
                            <div style="display:flex;align-items:center;gap:12px;">
                                <div style="width:72px;height:56px;border-radius:8px;background:{rrow['hex']};border:1px solid #ccc"></div>
                                <div style="flex:1;">
                                    <div style="font-weight:700;color:{accent}">{rrow['brand']}</div>
                                    <div style="font-size:14px;font-weight:600;margin-top:4px;">{rrow['name']}</div>
                                    <div style="color:#666;margin-top:6px;font-size:13px;">{rrow['product']}</div>
                                    <div style="color:#666;font-size:12px;margin-top:6px;">{(rrow['description'][:120] + '...') if pd.notna(rrow['description']) and len(str(rrow['description']))>120 else rrow['description']}</div>
                                </div>
                            </div>
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-top:10px;">
                                <div style="font-size:12px;color:#444">HEX: {rrow['hex']}</div>
                                <div>
                                    <a href="{rrow['url']}" target="_blank" style="background:{accent};color:#fff;padding:6px 10px;border-radius:8px;text-decoration:none;">View product</a>
                                </div>
                            </div>
                        </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

            # small debug / stats
            st.markdown("---")
            st.markdown(f"Found **{len(recs)}** matches (filtered). Closest match distance: **{recs['distance'].min():.3f}**")

# -----------------------------
# Footer / About
# -----------------------------
st.markdown("---")
st.markdown("""
**About**: Built using CIELAB color space + k-NN for perceptual matching.  
Tip: try toggling undertone/finish filters or selecting preferred brands to see tailored results.
""")
