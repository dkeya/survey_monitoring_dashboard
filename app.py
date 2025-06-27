# --- Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import altair as alt
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="SHAPe Avocado Dashboard",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stSlider [data-baseweb="slider"] {
            padding: 0;
        }
        .metric-card {
            border-radius: 10px;
            padding: 15px;
            background-color: #f0f2f6;
            margin-bottom: 15px;
        }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Data Loading Functions ---
@st.cache_data(ttl=3600)
def load_farmer_data():
    """Load the farmer baseline data"""
    try:
        df = pd.read_excel('shape_data.xlsx', sheet_name='Baseline')
        
        # Clean and preprocess data
        if 'data_time' in df.columns:
            df['submitdate'] = pd.to_datetime(df['data_time'], errors='coerce')
        
        # Handle area under cultivation
        area_col = '2.2 Total Area under Avocado Cultivation (Acres)'
        if area_col in df.columns:
            df['Total Area under Avocado Cultivation (Acres)'] = pd.to_numeric(df[area_col], errors='coerce')
        
        # Handle tree count
        trees_col = '2.3 Number of Avocado Trees Planted'
        if trees_col in df.columns:
            df['Number of Avocado Trees Planted'] = pd.to_numeric(df[trees_col], errors='coerce')
        
        # Calculate yields for different age groups if columns exist
        age_groups = {
            '0-3': '4.1 Average Yield per avocado Tree aged 0-3 years last season (kg)',
            '4-7': '4.11 Average Yield per avocado Tree aged 4-7 years last season (kg)',
            '8+': '4.12 Average Yield per avocado Tree aged 8+years last season (kg)'
        }
        
        for age, col in age_groups.items():
            if col in df.columns:
                df[f'Yield (kg/tree) {age} years'] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except Exception as e:
        st.error(f"Error loading farmer data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_exporter_metrics():
    """Load the exporter metrics data"""
    try:
        df = pd.read_excel('shape_data.xlsx', sheet_name='Metrics')
        
        # Check if required columns exist
        if 'Metrics' not in df.columns:
            st.warning("Metrics sheet doesn't have the expected structure")
            return pd.DataFrame()
            
        return df
    except Exception as e:
        st.warning(f"Couldn't load exporter metrics: {str(e)}")
        return pd.DataFrame()

# --- Visualization Functions ---
def create_farm_map(df):
    """Create interactive map of farms"""
    if df.empty or '_1.21 GPS Coordinates of Orchard_latitude' not in df.columns:
        return None
    
    # Filter valid coordinates
    map_df = df.dropna(subset=['_1.21 GPS Coordinates of Orchard_latitude', '_1.21 GPS Coordinates of Orchard_longitude'])
    
    if map_df.empty:
        return None
    
    # Create base map centered on Kenya
    m = folium.Map(location=[0.0236, 37.9062], zoom_start=6)
    
    # Add markers for each farm
    for idx, row in map_df.iterrows():
        popup_text = f"""
        <b>Farm:</b> {row.get('1.22 Orchard Name/ Name of farm', 'N/A')}<br>
        <b>Farmer:</b> {row.get('1.10 Farmer\'s Name (Three Names)', 'N/A')}<br>
        <b>Trees:</b> {row.get('2.3 Number of Avocado Trees Planted', 'N/A')}<br>
        <b>Variety:</b> {'Hass' if row.get('3.1 Variety Grown/Hass', 0) == 1 else 'Other'}
        """
        
        folium.Marker(
            location=[row['_1.21 GPS Coordinates of Orchard_latitude'], 
                     row['_1.21 GPS Coordinates of Orchard_longitude']],
            popup=folium.Popup(popup_text, max_width=250),
            icon=folium.Icon(color='green', icon='leaf')
        ).add_to(m)
    
    return m

def create_certification_chart(df):
    """Create certification status chart"""
    cert_cols = {
        'GlobalGAP': '6.2 Which Certifications is the Orchard compliant for this season?/Global GAP',
        'Organic': '6.2 Which Certifications is the Orchard compliant for this season?/Organic',
        'FairTrade': '6.2 Which Certifications is the Orchard compliant for this season?/FairTrade',
        'China': '1.26 General Administration of Customs of the Peoples Republic of China (GACC ) Approval Status'
    }
    
    cert_data = []
    for cert, col in cert_cols.items():
        if col in df.columns:
            if cert == 'China':
                count = (df[col] == 'Yes').sum()
            else:
                count = df[col].sum() if df[col].dtype in [int, float] else (df[col] == 1).sum()
            cert_data.append({'Certification': cert, 'Count': count})
    
    if not cert_data:
        return None
    
    cert_df = pd.DataFrame(cert_data)
    
    chart = alt.Chart(cert_df).mark_bar().encode(
        x='Certification:N',
        y='Count:Q',
        color=alt.Color('Certification:N', scale=alt.Scale(scheme='greens')),
        tooltip=['Certification', 'Count']
    ).properties(
        title='Farm Certification Status',
        width=600,
        height=400
    )
    
    return chart

def create_yield_comparison_chart(df):
    """Create yield comparison by tree age"""
    if df.empty:
        return None
    
    yield_data = []
    age_groups = {
        '0-3': '4.1 Average Yield per avocado Tree aged 0-3 years last season (kg)',
        '4-7': '4.11 Average Yield per avocado Tree aged 4-7 years last season (kg)',
        '8+': '4.12 Average Yield per avocado Tree aged 8+years last season (kg)'
    }
    
    for age, col in age_groups.items():
        if col in df.columns:
            avg_yield = df[col].mean()
            yield_data.append({'Age Group': f'{age} years', 'Average Yield (kg/tree)': avg_yield})
    
    if not yield_data:
        return None
    
    yield_df = pd.DataFrame(yield_data)
    
    chart = alt.Chart(yield_df).mark_bar().encode(
        x='Age Group:N',
        y='Average Yield (kg/tree):Q',
        color=alt.Color('Age Group:N', scale=alt.Scale(scheme='blues')),
        tooltip=['Age Group', 'Average Yield (kg/tree)']
    ).properties(
        title='Average Yield by Tree Age',
        width=600,
        height=400
    )
    
    return chart

def create_wordcloud(text, title):
    """Generate word cloud from text"""
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap='viridis').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16, pad=20)
    ax.axis('off')
    return fig

# --- Dashboard Sections ---
def show_overview(df, metrics_df):
    """Show overview/KPI cards"""
    st.subheader("Program Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Total farmers
    total_farmers = len(df)
    col1.markdown(f"""
    <div class="metric-card">
        <h3 style="margin:0; padding:0">Total Farmers</h3>
        <p style="margin:0; padding:0; font-size:24px">{total_farmers}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Total area under cultivation
    total_area = df['Total Area under Avocado Cultivation (Acres)'].sum()
    col2.markdown(f"""
    <div class="metric-card">
        <h3 style="margin:0; padding:0">Total Area (Acres)</h3>
        <p style="margin:0; padding:0; font-size:24px">{total_area:,.1f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Total trees
    total_trees = df['Number of Avocado Trees Planted'].sum()
    col3.markdown(f"""
    <div class="metric-card">
        <h3 style="margin:0; padding:0">Total Trees</h3>
        <p style="margin:0; padding:0; font-size:24px">{total_trees:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # GACC approved farms
    gacc_col = '1.26 General Administration of Customs of the Peoples Republic of China (GACC ) Approval Status'
    gacc_approved = df[gacc_col].eq('Yes').sum() if gacc_col in df.columns else 0
    col4.markdown(f"""
    <div class="metric-card">
        <h3 style="margin:0; padding:0">China-Approved Farms</h3>
        <p style="margin:0; padding:0; font-size:24px">{gacc_approved}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics comparison - only show if we have metrics data
    if not metrics_df.empty and 'Metrics' in metrics_df.columns:
        st.subheader("Progress vs Targets")
        
        # Get available metrics
        available_metrics = [m for m in ['# of Farms certified', '# of farmers', 
                                       'Land size under Hass avocado in Acres'] 
                           if m in metrics_df['Metrics'].values]
        
        if available_metrics:
            selected_metrics = st.multiselect(
                "Select metrics to compare",
                options=metrics_df['Metrics'].unique(),
                default=available_metrics[:3]  # Show first 3 by default
            )
            
            if selected_metrics:
                filtered_metrics = metrics_df[metrics_df['Metrics'].isin(selected_metrics)]
                
                # Check which periods we have data for
                periods = [p for p in ['Feb/Baseline', 'Total', 'Target'] 
                         if p in metrics_df.columns]
                
                if periods:
                    # Melt dataframe for Altair
                    melted_df = filtered_metrics.melt(id_vars='Metrics', 
                                                    value_vars=periods,
                                                    var_name='Period', 
                                                    value_name='Value')
                    
                    # Create chart
                    chart = alt.Chart(melted_df).mark_bar().encode(
                        x='Metrics:N',
                        y='Value:Q',
                        color='Period:N',
                        column='Period:N',
                        tooltip=['Metrics', 'Period', 'Value']
                    ).properties(
                        width=200,
                        height=400
                    )
                    
                    st.altair_chart(chart)
    else:
        st.info("Exporter metrics data not available or doesn't match expected format")

def show_geospatial(df):
    """Show farm location map"""
    st.subheader("Farm Locations")
    
    if not df.empty:
        m = create_farm_map(df)
        if m:
            folium_static(m, width=1000, height=600)
        else:
            st.warning("No valid geographic coordinates found in the data")
    else:
        st.warning("No data available for mapping")

def show_certification(df):
    """Show certification status"""
    st.subheader("Certification Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cert_chart = create_certification_chart(df)
        if cert_chart:
            st.altair_chart(cert_chart, use_container_width=True)
        else:
            st.warning("No certification data available")
    
    with col2:
        # Show certification requirements checklist
        st.markdown("**China Market Requirements Checklist**")
        
        requirements = [
            ("Farm registration with KEPHIS", "1.25 KEPHIS Registration Status"),
            ("GACC approval", "1.26 General Administration of Customs of the Peoples Republic of China (GACC ) Approval Status"),
            ("Pest monitoring records", "3.81 Pest Monitoring"),
            ("Sanitation records", "3.6 Is there a record of sanitation conditions?"),
            ("Approved pesticide use", "6.7 Use of Approved Pesticides Only")
        ]
        
        for req, col in requirements:
            if col in df.columns:
                compliant = df[col].eq('Yes').sum()
                total = len(df)
                st.progress(compliant/total, text=f"{req}: {compliant}/{total} farms")
            else:
                st.text(f"{req}: Data not available")

def show_production_metrics(df):
    """Show production metrics"""
    st.subheader("Production Metrics")
    
    tab1, tab2, tab3 = st.tabs(["Yields", "Inputs", "Losses"])
    
    with tab1:
        yield_chart = create_yield_comparison_chart(df)
        if yield_chart:
            st.altair_chart(yield_chart, use_container_width=True)
        else:
            st.warning("No yield data available")
        
        # Add variety distribution
        if '3.1 Variety Grown/Hass' in df.columns:
            varieties = ['Hass', 'Fuerte', 'Pinkerton', 'Other']
            variety_counts = {v: df[f'3.1 Variety Grown/{v}'].sum() for v in varieties if f'3.1 Variety Grown/{v}' in df.columns}
            
            if variety_counts:
                variety_df = pd.DataFrame.from_dict(variety_counts, orient='index', columns=['Count']).reset_index()
                variety_df.columns = ['Variety', 'Count']
                
                variety_chart = alt.Chart(variety_df).mark_arc().encode(
                    theta='Count:Q',
                    color='Variety:N',
                    tooltip=['Variety', 'Count']
                ).properties(
                    title='Avocado Varieties Grown',
                    width=400,
                    height=400
                )
                
                st.altair_chart(variety_chart)
    
    with tab2:
        # Input usage visualization
        if '3.3 Type of Fertilizer Used/Organic' in df.columns:
            fertilizer_data = {
                'Type': ['Organic', 'Inorganic', 'None'],
                'Count': [
                    df['3.3 Type of Fertilizer Used/Organic'].sum(),
                    df['3.3 Type of Fertilizer Used/Inorganic'].sum(),
                    df['3.3 Type of Fertilizer Used/None'].sum()
                ]
            }
            
            fertilizer_df = pd.DataFrame(fertilizer_data)
            
            fertilizer_chart = alt.Chart(fertilizer_df).mark_bar().encode(
                x='Type:N',
                y='Count:Q',
                color='Type:N',
                tooltip=['Type', 'Count']
            ).properties(
                title='Fertilizer Usage',
                width=600,
                height=400
            )
            
            st.altair_chart(fertilizer_chart)
        
        # Irrigation practices
        if '3.5 Irrigation Practices/Rainfed' in df.columns:
            irrigation_data = {
                'Method': ['Rainfed', 'Manual', 'Drip', 'Sprinkler'],
                'Count': [
                    df['3.5 Irrigation Practices/Rainfed'].sum(),
                    df['3.5 Irrigation Practices/Manual Watering'].sum(),
                    df['3.5 Irrigation Practices/Drip'].sum() if '3.5 Irrigation Practices/Drip' in df.columns else 0,
                    df['3.5 Irrigation Practices/Sprinkler'].sum() if '3.5 Irrigation Practices/Sprinkler' in df.columns else 0
                ]
            }
            
            irrigation_df = pd.DataFrame(irrigation_data)
            
            irrigation_chart = alt.Chart(irrigation_df).mark_bar().encode(
                x='Method:N',
                y='Count:Q',
                color='Method:N',
                tooltip=['Method', 'Count']
            ).properties(
                title='Irrigation Methods',
                width=600,
                height=400
            )
            
            st.altair_chart(irrigation_chart)
    
    with tab3:
        # Post-harvest losses
        if '4.3 Avocado Losses last season (kg)' in df.columns:
            loss_reasons = {
                'Bruising': '4.32 Other Causes of Loss last season/Bruising',
                'Disease': '4.32 Other Causes of Loss last season/Disease',
                'Poor Handling': '4.32 Other Causes of Loss last season/Poor Handling',
                'Theft': '4.32 Other Causes of Loss last season/Theft'
            }
            
            loss_data = []
            for reason, col in loss_reasons.items():
                if col in df.columns:
                    loss_data.append({
                        'Reason': reason,
                        'Count': df[col].sum()
                    })
            
            if loss_data:
                loss_df = pd.DataFrame(loss_data)
                
                loss_chart = alt.Chart(loss_df).mark_bar().encode(
                    x='Count:Q',
                    y='Reason:N',
                    color='Reason:N',
                    tooltip=['Reason', 'Count']
                ).properties(
                    title='Primary Causes of Post-Harvest Loss',
                    width=600,
                    height=400
                )
                
                st.altair_chart(loss_chart)

def show_market_analysis(df):
    """Show market and income analysis"""
    st.subheader("Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market outlets
        if '5.1 Main Market Outlet' in df.columns:
            market_counts = df['5.1 Main Market Outlet'].value_counts().reset_index()
            market_counts.columns = ['Outlet', 'Count']
            
            market_chart = alt.Chart(market_counts).mark_bar().encode(
                x='Count:Q',
                y='Outlet:N',
                color='Outlet:N',
                tooltip=['Outlet', 'Count']
            ).properties(
                title='Main Market Outlets',
                width=400,
                height=400
            )
            
            st.altair_chart(market_chart)
    
    with col2:
        # Income by farm size
        if '5.3 Total Income from Avocado Sales (KSH last season)' in df.columns and '2.1 Total Farm Size (Acres)' in df.columns:
            df['Farm Size Category'] = pd.cut(df['2.1 Total Farm Size (Acres)'],
                                            bins=[0, 3, 10, float('inf')],
                                            labels=['Small (<3 acres)', 'Medium (3-10 acres)', 'Large (>10 acres)'])
            
            income_df = df.groupby('Farm Size Category')['5.3 Total Income from Avocado Sales (KSH last season)'].mean().reset_index()
            
            income_chart = alt.Chart(income_df).mark_bar().encode(
                x='Farm Size Category:N',
                y='5.3 Total Income from Avocado Sales (KSH last season):Q',
                color='Farm Size Category:N',
                tooltip=['Farm Size Category', '5.3 Total Income from Avocado Sales (KSH last season)']
            ).properties(
                title='Average Income by Farm Size',
                width=400,
                height=400
            )
            
            st.altair_chart(income_chart)
    
    # Challenges word cloud
    if '5.10 Challenges in Market Access/Quality Standards' in df.columns:
        challenges_text = ""
        challenge_cols = [
            '5.10 Challenges in Market Access/Price Fluctuations',
            '5.10 Challenges in Market Access/Limited Buyers',
            '5.10 Challenges in Market Access/Quality Standards',
            '5.10 Challenges in Market Access/Other'
        ]
        
        for col in challenge_cols:
            if col in df.columns and df[col].sum() > 0:
                challenge_name = col.split('/')[-1]
                challenges_text += (challenge_name + ' ') * int(df[col].sum())
        
        if challenges_text:
            fig = create_wordcloud(challenges_text, "Market Access Challenges")
            st.pyplot(fig)

def show_training_needs(df):
    """Show training and extension needs"""
    st.subheader("Training & Extension Needs")
    
    if '8.6 What are your most pressing training/extension needs/GAP' in df.columns:
        needs_data = {
            'Need': ['GAP', 'Post-Harvest', 'Certification', 'Market Access'],
            'Count': [
                df['8.6 What are your most pressing training/extension needs/GAP'].sum(),
                df['8.6 What are your most pressing training/extension needs/Post-Harvest Management'].sum(),
                df['8.6 What are your most pressing training/extension needs/Certification Compliance'].sum(),
                df['8.6 What are your most pressing training/extension needs/Market Access'].sum()
            ]
        }
        
        needs_df = pd.DataFrame(needs_data)
        
        needs_chart = alt.Chart(needs_df).mark_bar().encode(
            x='Count:Q',
            y='Need:N',
            color='Need:N',
            tooltip=['Need', 'Count']
        ).properties(
            title='Most Pressing Training Needs',
            width=600,
            height=400
        )
        
        st.altair_chart(needs_chart)
    
    # Suggestions word cloud
    if 'Suggestions for the Shape Program Improvement' in df.columns:
        suggestions = ' '.join(df['Suggestions for the Shape Program Improvement'].dropna().astype(str))
        
        if suggestions.strip():
            fig = create_wordcloud(suggestions, "Farmer Suggestions for Program Improvement")
            st.pyplot(fig)

def main():
    st.title("🥑 SHAPe Avocado Dashboard")
    st.markdown("Monitoring Kenya's avocado value chain for export excellence")
    
    # Load data with error handling
    try:
        farmer_df = load_farmer_data()
        metrics_df = load_exporter_metrics()
        
        if farmer_df.empty:
            st.warning("No farmer data loaded. Please check your data file.")
            return
    
        # Sidebar filters
        st.sidebar.title("Filters")
        
        # Company filter
        companies = farmer_df['1.1 Company Name'].unique()
        selected_company = st.sidebar.selectbox(
            "Select Exporter",
            options=['All'] + list(companies),
            index=0
        )
        
        # Date filter
        if not farmer_df['submitdate'].isna().all():
            min_date = farmer_df['submitdate'].min().date()
            max_date = farmer_df['submitdate'].max().date()
            
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=[min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
        
        # Apply filters
        filtered_df = farmer_df.copy()
        
        if selected_company != 'All':
            filtered_df = filtered_df[filtered_df['1.1 Company Name'] == selected_company]
        
        if not farmer_df['submitdate'].isna().all() and len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['submitdate'].dt.date >= date_range[0]) & 
                (filtered_df['submitdate'].dt.date <= date_range[1])
            ]
        
        # Dashboard sections
        show_overview(filtered_df, metrics_df)
        show_geospatial(filtered_df)
        show_certification(filtered_df)
        show_production_metrics(filtered_df)
        show_market_analysis(filtered_df)
        show_training_needs(filtered_df)
        
        # Data explorer
        st.subheader("Data Explorer")
        if st.checkbox("Show raw data"):
            st.dataframe(filtered_df)
        
        # Download button
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df(filtered_df)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name='shape_filtered_data.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please ensure: \n1. Your data file is named 'shape_data.xlsx' \n2. It contains 'Baseline' and 'Metrics' sheets \n3. The columns match the expected format")

if __name__ == "__main__":
    main()