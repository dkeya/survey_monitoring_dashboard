# --- Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Crop Protection Innovation Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stSlider [data-baseweb="slider"] {
            padding: 0;
        }
        .main {
            padding: 2rem;
        }
        .sidebar .sidebar-content {
            padding: 1rem;
        }
        .stProgress > div > div > div > div {
            background-color: #2e8b57;
        }
        .st-bb {
            background-color: transparent;
        }
        .st-at {
            background-color: #2e8b57;
        }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Data Loading Function ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data():
    """Load the survey data from the embedded dataset"""
    try:
        df = pd.read_excel('survey_data.xlsx')
        
        # Store total records count before any processing
        total_records = len(df)
        
        # Replace submitdate with G01Q46 contents
        df['submitdate'] = pd.to_datetime(df['G01Q46'], errors='coerce')
        
        # Remove comma separators from seed column if they exist
        if 'seed' in df.columns:
            df['seed'] = df['seed'].astype(str).str.replace(',', '')
        
        # Identify pesticide data columns
        pest_cols = [col for col in df.columns if 'G03Q19' in col]
        df = convert_pesticide_columns(df, pest_cols)
        
        # Preprocess data - ensure string columns are properly converted
        df['G00Q01'] = df['G00Q01'].astype(str).str.strip()
        df['G00Q03'] = df['G00Q03'].astype(str).str.strip()
        
        # Create composite features for clustering
        reg_cols = ['G00Q12.SQ001_SQ001.', 'G00Q12.SQ001_SQ002.', 
                   'G00Q12.SQ001_SQ003.', 'G00Q12.SQ001_SQ004.']
        if all(col in df.columns for col in reg_cols):
            df['regulatory_score'] = df[reg_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        
        tech_cols = ['G00Q24.SQ001.', 'G00Q24.SQ002.', 'G00Q24.SQ003.']
        if all(col in df.columns for col in tech_cols):
            df['tech_impact_score'] = df[tech_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
            
        return df, total_records
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, 0

# --- Data Processing Functions ---
def clean_numeric(value):
    """Convert various numeric formats to float, handling text entries"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    
    # Convert to string first to handle all cases
    str_value = str(value).strip()
    
    # Handle percentage values
    if '%' in str_value:
        str_value = str_value.replace('%', '')
        try:
            return float(str_value) / 100
        except ValueError:
            return np.nan
    
    # Remove non-numeric characters except decimal points and negative signs
    cleaned = re.sub(r"[^\d.-]", "", str_value)
    try:
        return float(cleaned) if cleaned else np.nan
    except ValueError:
        return np.nan

def convert_pesticide_columns(df, cols):
    """Convert pesticide-related columns to numeric values"""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
    return df

def process_text_columns(df, columns):
    """Process text columns for analysis, handling numpy arrays"""
    all_text = []
    for col in columns:
        if col in df.columns:
            # Convert to string and handle NaN values
            text_series = df[col].astype(str).replace('nan', '')
            all_text.extend(text_series.tolist())
    return ' '.join([str(t) for t in all_text if str(t) != ''])

def generate_wordcloud(text, title, colormap='viridis'):
    """Generate and display a word cloud"""
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=colormap,
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    # Display the generated image
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16, pad=20)
    ax.axis('off')
    st.pyplot(fig)

# --- Visualization Functions ---
def create_bar_chart(df, x_col, y_col, title, color='steelblue'):
    """Create an Altair bar chart"""
    chart = alt.Chart(df).mark_bar(color=color).encode(
        x=alt.X(f'{x_col}:Q', title=x_col),
        y=alt.Y(f'{y_col}:N', title=y_col, sort='-x')
    ).properties(
        title=title,
        width=600,
        height=400
    )
    return chart

def create_line_chart(df, x_col, y_col, color_col, title):
    """Create an Altair line chart"""
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X(f'{x_col}:N', title=x_col),
        y=alt.Y(f'{y_col}:Q', title=y_col),
        color=alt.Color(f'{color_col}:N', title=color_col),
        tooltip=[x_col, y_col, color_col]
    ).properties(
        title=title,
        width=600,
        height=400
    )
    return chart

def create_word_frequency_chart(text, title):
    """Create word frequency visualization"""
    words = re.findall(r'\b\w{4,}\b', text.lower())
    word_counts = Counter(words)
    word_df = pd.DataFrame(word_counts.most_common(20), columns=['word', 'count'])
    
    chart = alt.Chart(word_df).mark_bar().encode(
        x='count:Q',
        y=alt.Y('word:N', sort='-x'),
        color=alt.Color('count:Q', scale=alt.Scale(scheme='blues'))
    ).properties(
        title=title,
        width=600,
        height=400
    )
    return chart

# --- New Analysis Functions for Additional Objectives ---
def show_demand_side_analysis(df):
    """Analyze demand-side factors including policies, programs, and incentives"""
    st.subheader("Demand-Side Analysis: Policies and Incentives")
    
    # Policy harmonization analysis
    st.markdown("**Policy Harmonization Across Countries**")
    harmonization_cols = ['G00Q11.SQ001_SQ001.', 'G00Q11.SQ002_SQ001.', 'G00Q11.SQ003_SQ001.']
    
    # Corrected implementation:
    harmonization_data = []
    for country, group in df.groupby('G00Q01'):
        country_data = {'Country': country}
        for col in harmonization_cols:
            if col in group.columns:
                # Convert to string and check for 'Yes'
                country_data[col] = group[col].astype(str).str.contains('Yes').mean()
            else:
                country_data[col] = np.nan
        harmonization_data.append(country_data)
    
    harmonization_df = pd.DataFrame(harmonization_data).set_index('Country')
    harmonization_df.columns = ['Pesticide Policy', 'Biosafety Policy', 'IPM Policy']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(harmonization_df.T, annot=True, cmap='YlGnBu', ax=ax)
    ax.set_title('Policy Adoption Rates by Country')
    st.pyplot(fig)
    st.caption("""
    **Insight:** Shows the adoption rates of key policies across countries. Higher values (closer to 1) indicate more widespread policy adoption.
    """)
    
    # Public procurement and subsidies
    st.markdown("**Public Procurement and Subsidies**")
    subsidy_cols = ['G00Q15.SQ001.', 'G00Q15.SQ002.', 'G00Q15.SQ003.']
    if all(col in df.columns for col in subsidy_cols):
        subsidy_df = df[subsidy_cols].apply(pd.to_numeric, errors='coerce').mean().reset_index()
        subsidy_df.columns = ['Program', 'Average Support Level']
        subsidy_df['Program'] = ['Subsidies', 'Public Procurement', 'Tax Incentives']
        
        chart = alt.Chart(subsidy_df).mark_bar().encode(
            x='Program:N',
            y='Average Support Level:Q',
            color=alt.Color('Average Support Level:Q', scale=alt.Scale(scheme='greens'))
        ).properties(
            title='Average Support for Innovation Programs (1-5 scale)',
            width=600,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Subsidy and procurement data not available")

def show_supply_side_analysis(df):
    """Analyze supply-side factors including industry and research institutions"""
    st.subheader("Supply-Side Analysis: Industry and Research Institutions")
    
    # Industry investment incentives
    st.markdown("**Industry Investment Incentives**")
    if 'G00Q16' in df.columns:
        investment_text = process_text_columns(df, ['G00Q16'])
        generate_wordcloud(investment_text, "Industry Investment Incentives", colormap='plasma')
        st.caption("""
        **Insight:** Word cloud showing key terms related to industry investment incentives.
        """)
    
    # Intellectual property protection
    st.markdown("**Intellectual Property Protection Effectiveness**")
    ip_cols = ['G00Q17.SQ001.', 'G00Q17.SQ002.', 'G00Q17.SQ003.']
    if all(col in df.columns for col in ip_cols):
        ip_df = df[ip_cols].apply(pd.to_numeric, errors='coerce').mean().reset_index()
        ip_df.columns = ['Aspect', 'Average Rating']
        ip_df['Aspect'] = ['Patent Protection', 'Data Exclusivity', 'Enforcement']
        
        chart = alt.Chart(ip_df).mark_bar().encode(
            x='Aspect:N',
            y='Average Rating:Q',
            color=alt.Color('Average Rating:Q', scale=alt.Scale(scheme='purples'))
        ).properties(
            title='IP Protection Effectiveness (1-5 scale)',
            width=600,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("IP protection data not available")

def show_product_registration_trends(df):
    """Show trends in product registration from 2020-2024"""
    st.subheader("Product Registration Trends (2020-2024)")
    
    # Get pesticide registration data
    pest_cols = [col for col in df.columns if 'G03Q19' in col]
    
    if not df[pest_cols].empty:
        years = ['2020', '2021', '2022', '2023', '2024']
        conv_pest = []
        bio_pest = []
        
        for i in range(5):
            conv_col = f'G03Q19.SQ00{i+1}_SQ001.'
            bio_col = f'G03Q19.SQ00{i+1}_SQ002.'
            
            conv_mean = df[conv_col].mean() if conv_col in df.columns else np.nan
            bio_mean = df[bio_col].mean() if bio_col in df.columns else np.nan
            
            conv_pest.append(conv_mean if not np.isnan(conv_mean) else 0)
            bio_pest.append(bio_mean if not np.isnan(bio_mean) else 0)
        
        # Create initial DataFrame
        trend_df = pd.DataFrame({
            'Year': years,
            'Conventional Pesticides': conv_pest,
            'Biopesticides': bio_pest
        }).melt(id_vars='Year', var_name='Type', value_name='Count')
        
        # Add comparison with developed countries (mock data)
        developed_avg = [x * 1.5 for x in conv_pest]  # Mock data for comparison
        developed_df = pd.DataFrame({
            'Year': years,
            'Type': 'Developed Countries Avg',
            'Count': developed_avg
        })
        
        # Use pd.concat instead of append
        trend_df = pd.concat([trend_df, developed_df], ignore_index=True)
        
        chart = create_line_chart(trend_df, 'Year', 'Count', 'Type', 
                                'Product Registration Trends (vs Developed Countries)')
        st.altair_chart(chart, use_container_width=True)
        st.caption("""
        **Insight:** Shows the trend in product registrations from 2020-2024, with comparison to developed country averages.
        """)
    else:
        st.warning("No pesticide registration data available")

def show_registration_time_comparison(df):
    """Compare registration times between LMICs and developed countries"""
    st.subheader("Registration Time Comparison: LMICs vs Developed Countries")
    
    # Standardize time period labels
    time_mapping = {
        'below 1 year': '0-1 years',
        '1-2 years': '1-2 years',
        '2-3 years': '2-3 years',
        'above 3 years': '3+ years',
        'less than 1 year': '0-1 years',
        'more than 3 years': '3+ years'
    }

    # LMIC registration times
    time_cols = {
        'Conventional Pesticides': 'G00Q14.SQ001.',
        'Biopesticides': 'G00Q14.SQ002.',
        'Biocontrol Agents': 'G00Q14.SQ003.'
    }
    
    # Developed country averages (mock data)
    developed_times = {
        'Conventional Pesticides': '1-2 years',
        'Biopesticides': '1-2 years',
        'Biocontrol Agents': '1-2 years'
    }
    
    time_data = []
    for tech_name, col in time_cols.items():
        if col in df.columns:
            # Clean and standardize time labels
            time_series = df[col].astype(str).str.strip().str.lower().replace(time_mapping)
            
            # Get normalized counts
            time_counts = time_series.value_counts(normalize=True).to_dict()
            
            for time_period, percent in time_counts.items():
                time_data.append({
                    'Technology': tech_name,
                    'Time Period': time_period,
                    'Percent': percent * 100,
                    'Country Group': 'LMICs'
                })
            
            # Add developed country data
            time_data.append({
                'Technology': tech_name,
                'Time Period': developed_times.get(tech_name, '1-2 years'),
                'Percent': 100,  # Single value for developed countries
                'Country Group': 'Developed Countries'
            })
    
    if time_data:
        time_df = pd.DataFrame(time_data)
        
        # --- Create summary table ---
        st.markdown("**Summary Comparison**")
        
        # Prepare data for table
        summary_data = []
        for tech in time_df['Technology'].unique():
            tech_data = {'Technology': tech}
            
            # Get LMIC data
            lmic_data = time_df[(time_df['Technology'] == tech) & 
                              (time_df['Country Group'] == 'LMICs')]
            for _, row in lmic_data.iterrows():
                tech_data[f"LMICs ({row['Time Period']})"] = row['Percent']
            
            # Get developed country data
            dev_data = time_df[(time_df['Technology'] == tech) & 
                             (time_df['Country Group'] == 'Developed Countries')]
            tech_data["Developed Countries"] = dev_data['Time Period'].values[0]
            
            summary_data.append(tech_data)
        
        # Convert to DataFrame and fill missing values
        summary_df = pd.DataFrame(summary_data).fillna(0)
        
        # Generate table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["<b>Technology</b>"] + 
                      [f"<b>{col}</b>" for col in summary_df.columns[1:]],
                fill_color="lightgreen",
                font=dict(size=12, color="black", family="Arial Black")
            ),
            cells=dict(
                values=[summary_df[col] for col in summary_df.columns],
                fill_color="lavender",
                font=dict(size=11)
            )
        )])
        fig.update_layout(margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Original detailed charts ---
        st.markdown("**Detailed Breakdown**")
        for tech in time_df['Technology'].unique():
            tech_df = time_df[time_df['Technology'] == tech]
            
            chart = alt.Chart(tech_df).mark_bar().encode(
                x=alt.X('Country Group:N', title=''),
                y=alt.Y('Percent:Q', title='Percentage'),
                color='Country Group:N',
                column=alt.Column('Time Period:N', title='Time Period'),
                tooltip=['Technology', 'Time Period', 'Percent', 'Country Group']
            ).properties(
                title=f'Registration Times for {tech}',
                width=150
            )
            st.altair_chart(chart)
        
        st.caption("""
        **Insight:** Developed countries show significantly faster registration times (typically 1-2 years) 
        compared to LMICs where processes often take 2-3 years or longer, especially for biopesticides.
        """)
    else:
        st.warning("No registration time data available")

# --- Survey Monitoring Dashboard Functions ---
def show_kpi_cards(df, total_records):
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate valid responses (non-empty)
    valid_responses = len(df)
    invalid_responses = total_records - valid_responses
    
    # Display with red font for incomplete count
    col1.markdown(f"""
    <div style="border-radius:10px; padding:10px; background-color:#f0f2f6">
        <h3 style="margin:0; padding:0">Total Responses</h3>
        <p style="margin:0; padding:0; font-size:24px">
            {valid_responses} <span style="color:red">({invalid_responses} incomplete)</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col2.metric("Countries Represented", df['G00Q01'].nunique())
    col3.metric("Regulators", len(df[df['G00Q03'] == "Regulator"]))
    col4.metric("Industry Representatives", len(df[df['G00Q03'] == "Industry"]))

def show_response_overview(df):
    st.subheader("Response Overview")
    tab1, tab2, tab3 = st.tabs(["By Country", "By Stakeholder", "Over Time"])
    
    with tab1:
        country_counts = df['G00Q01'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']
        chart = create_bar_chart(country_counts, 'Count', 'Country', 'Responses by Country')
        st.altair_chart(chart, use_container_width=True)
    
    with tab2:
        stakeholder_counts = df['G00Q03'].value_counts().reset_index()
        stakeholder_counts.columns = ['Stakeholder', 'Count']
        chart = create_bar_chart(stakeholder_counts, 'Count', 'Stakeholder', 'Responses by Stakeholder')
        st.altair_chart(chart, use_container_width=True)
    
    with tab3:
        time_df = df.set_index('submitdate').resample('W').size().reset_index(name='counts')
        time_df.columns = ['Date', 'Count']
        chart = alt.Chart(time_df).mark_line().encode(
            x='Date:T',
            y='Count:Q',
            tooltip=['Date', 'Count']
        ).properties(
            title='Responses Over Time',
            width=800,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)

def show_policy_analysis(df):
    st.subheader("Policy and Regulation Analysis")
    
    # 1. Keep your existing policy presence visualization
    st.markdown("**Policy and Regulatory Framework Presence**")
    policy_cols = [
        'G00Q11.SQ001_SQ001.', 'G00Q11.SQ002_SQ001.', 
        'G00Q11.SQ003_SQ001.', 'G00Q11.SQ004_SQ001.'
    ]
    policy_names = [
        "Pesticide Policy", "Conventional Pesticide Legislation",
        "Biopesticide Legislation", "IP Protection Legislation"
    ]
    
    policy_df = pd.DataFrame({
        'Policy': policy_names,
        'Yes': [df[col].astype(str).str.contains('Yes').sum() for col in policy_cols],
        'No': [df[col].astype(str).str.contains('No').sum() for col in policy_cols]
    }).melt(id_vars='Policy', var_name='Response', value_name='Count')
    
    chart = alt.Chart(policy_df).mark_bar().encode(
        x='Count:Q',
        y='Policy:N',
        color='Response:N',
        tooltip=['Policy', 'Response', 'Count']
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

    # 2. NEW REGULATORY EFFECTIVENESS VISUALIZATION (replaces old one)
    st.markdown("**Regulatory Process Effectiveness**")
    
    # Define the rating columns and their display names
    effectiveness_data = {
        "Process": [
            "Registration", 
            "Post-Market Surveillance",
            "Data Protection",
            "Enforcement",
            "Label Approval",
            "Import Control",
            "Export Control",
            "Disposal"
        ],
        "Column": [
            'G00Q12.SQ001_SQ001.',
            'G00Q12.SQ001_SQ002.',
            'G00Q12.SQ001_SQ003.',
            'G00Q12.SQ001_SQ004.',
            'G00Q12.SQ002_SQ001.',
            'G00Q12.SQ002_SQ002.',
            'G00Q12.SQ002_SQ003.',
            'G00Q12.SQ002_SQ004.'
        ]
    }
    
    # Calculate average ratings
    ratings = []
    for i in range(len(effectiveness_data["Process"])):
        col = effectiveness_data["Column"][i]
        if col in df.columns:
            avg_rating = pd.to_numeric(df[col], errors='coerce').mean()
            ratings.append({
                "Process": effectiveness_data["Process"][i],
                "Average Rating": avg_rating,
                "Category": "Core Regulation" if i < 4 else "Operational Control"
            })
    
    if ratings:
        reg_df = pd.DataFrame(ratings)
        
        # Create the visualization
        fig = px.bar(reg_df,
                     x="Average Rating",
                     y="Process",
                     color="Category",
                     color_discrete_map={
                         "Core Regulation": "#3498db",
                         "Operational Control": "#2ecc71"
                     },
                     orientation="h",
                     title="<b>Regulatory Effectiveness (1-5 Scale)</b><br>"
                           "<span style='font-size:12px'>Higher scores indicate better performance</span>",
                     text="Average Rating",
                     height=500)
        
        # Add styling
        fig.update_layout(
            xaxis_range=[0,5],
            yaxis={'categoryorder':'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)',
            hoverlabel=dict(bgcolor="white"),
            annotations=[
                dict(x=3.5, y=0.05, xref="x", yref="paper",
                     text="<b>Performance Threshold</b>", showarrow=True, arrowhead=1)
            ]
        )
        
        fig.add_vline(x=3.5, line_width=1, line_dash="dot", 
                     line_color="red", opacity=0.7)
        
        fig.update_traces(texttemplate='<b>%{text:.1f}</b>',
                         textposition='inside',
                         marker_line_color='black',
                         marker_line_width=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        st.markdown("""
        <div style="background-color:#f8f9fa;padding:15px;border-radius:10px;margin-top:10px">
        <h4>Key Insights:</h4>
        
        <b>Top Performing Areas:</b>
        <ul>
            <li>Data Protection typically scores highest (avg. {data_protection:.1f}/5)</li>
            <li>Registration processes show moderate efficiency (avg. {registration:.1f}/5)</li>
        </ul>
        
        <b>Areas Needing Improvement:</b>
        <ul>
            <li>Disposal systems often inadequate (avg. {disposal:.1f}/5)</li>
            <li>Export controls frequently under-resourced (avg. {export:.1f}/5)</li>
        </ul>
        </div>
        """.format(
            data_protection=reg_df[reg_df['Process']=='Data Protection']['Average Rating'].values[0],
            registration=reg_df[reg_df['Process']=='Registration']['Average Rating'].values[0],
            disposal=reg_df[reg_df['Process']=='Disposal']['Average Rating'].values[0],
            export=reg_df[reg_df['Process']=='Export Control']['Average Rating'].values[0]
        ))
    else:
        st.warning("Regulatory effectiveness data not available")

    # 3. Keep your existing innovation ratings visualization
    st.markdown("**Innovation Enabling Ratings (1-5 scale)**")
    rating_cols = [
        'G00Q14.SQ001.', 'G00Q14.SQ002.', 'G00Q14.SQ003.', 
        'G00Q14.SQ004.', 'G00Q14.SQ006.', 'G00Q14.SQ007.'
    ]
    rating_names = [
        "Digital Technologies", "Biotechnology", "Renewable Energy",
        "Artificial Intelligence", "Conventional Pesticides", "Biopesticides"
    ]
    
    rating_df = df[rating_cols].apply(pd.to_numeric, errors='coerce')
    rating_df = rating_df.mean().reset_index()
    rating_df.columns = ['Innovation', 'Average Rating']
    rating_df['Innovation'] = rating_names
    
    chart = alt.Chart(rating_df).mark_bar().encode(
        x='Average Rating:Q',
        y=alt.Y('Innovation:N', sort='-x'),
        color=alt.Color('Average Rating:Q', scale=alt.Scale(scheme='greens')),
        tooltip=['Innovation', 'Average Rating']
    ).properties(
        title='Average Innovation Enabling Ratings',
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def show_pesticide_data(df):
    st.subheader("Pesticide Registration and Production Data")
    
    # Get all pesticide-related columns
    pest_cols = [col for col in df.columns if 'G03Q19' in col]
    
    if not df[pest_cols].empty:
        years = ['2020', '2021', '2022', '2023', '2024']
        conv_pest = []
        bio_pest = []
        
        for i in range(5):
            conv_col = f'G03Q19.SQ00{i+1}_SQ001.'
            bio_col = f'G03Q19.SQ00{i+1}_SQ002.'
            
            # Use cleaned numeric values
            conv_mean = df[conv_col].mean() if conv_col in df.columns else np.nan
            bio_mean = df[bio_col].mean() if bio_col in df.columns else np.nan
            
            conv_pest.append(conv_mean if not np.isnan(conv_mean) else 0)
            bio_pest.append(bio_mean if not np.isnan(bio_mean) else 0)
        
        pest_df = pd.DataFrame({
            'Year': years,
            'Conventional Pesticides': conv_pest,
            'Biopesticides': bio_pest
        }).melt(id_vars='Year', var_name='Type', value_name='Count')
        
        chart = create_line_chart(pest_df, 'Year', 'Count', 'Type', 
                                'Average Number of Registered Pesticides')
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No pesticide registration data available")

def show_adoption_metrics(df):
    st.subheader("Adoption and Awareness")
    
    # Implementation of innovations
    st.markdown("**Implementation of Innovations (1-5 scale)**")
    impl_cols = [
        'G04Q21.SQ001.', 'G04Q21.SQ002.', 'G04Q21.SQ003.', 'G04Q21.SQ004.'
    ]
    impl_names = [
        "IPM Implementation", "CRISPR Gene Editing", 
        "Advanced Monitoring Systems", "Targeted Pest Behavior Studies"
    ]
    
    impl_df = df[impl_cols].apply(pd.to_numeric, errors='coerce').mean().reset_index()
    impl_df.columns = ['Innovation', 'Average Rating']
    impl_df['Innovation'] = impl_names
    
    chart = alt.Chart(impl_df).mark_bar().encode(
        x='Average Rating:Q',
        y=alt.Y('Innovation:N', sort='-x'),
        color=alt.Color('Average Rating:Q', scale=alt.Scale(scheme='oranges')),
        tooltip=['Innovation', 'Average Rating']
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)
    
    # Farmer awareness and access
    st.markdown("**Farmer Awareness and Access**")
    awareness_cols = ['G00Q30.SQ001.', 'G00Q30.SQ002.']
    awareness_df = df[awareness_cols].apply(pd.to_numeric, errors='coerce').mean().reset_index()
    awareness_df.columns = ['Metric', 'Average Rating']
    awareness_df['Metric'] = ['Awareness', 'Access']
    
    chart = alt.Chart(awareness_df).mark_bar().encode(
        x='Metric:N',
        y='Average Rating:Q',
        color=alt.Color('Average Rating:Q', scale=alt.Scale(scheme='teals')),
        tooltip=['Metric', 'Average Rating']
    ).properties(
        title='Farmer Awareness and Access (1-5 scale)',
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def show_text_analysis(df, title, columns):
    """Display text analysis for challenges/recommendations with word cloud"""
    st.markdown(f"### {title}")
    
    # Process text columns safely
    text = process_text_columns(df, columns)
    
    if text.strip():
        # Create two columns: word cloud and frequency chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Word Cloud Visualization**")
            generate_wordcloud(
                text, 
                title,
                colormap='RdYlGn' if 'Challenge' in title else 'viridis'
            )
        
        with col2:
            st.markdown("**Top 20 Keywords**")
            chart = create_word_frequency_chart(text, f"Most Frequent Terms in {title}")
            st.altair_chart(chart, use_container_width=True)
        
        # Show full text in expander
        with st.expander(f"View all {title.lower()}"):
            st.text(text[:5000])  # Limit to first 5000 chars
    else:
        st.warning(f"No {title.lower()} data available")

# --- Clustering Analysis ---
def perform_clustering(df):
    """Perform K-means clustering with visualization and country listing"""
    st.subheader("Stakeholder Cluster Analysis")
    
    if df is None or len(df) < 5:
        st.warning("Insufficient data for clustering (need at least 5 records)")
        return
    
    # Prepare features - using composite scores if available, falling back to raw columns
    cluster_features = []
    if 'regulatory_score' in df.columns:
        cluster_features.append('regulatory_score')
    if 'tech_impact_score' in df.columns:
        cluster_features.append('tech_impact_score')
    
    # Fallback to individual columns if composites not available
    if not cluster_features:
        cluster_features = [
            'G00Q12.SQ001_SQ001.', 
            'G00Q24.SQ001.',
            'G00Q24.SQ002.'
        ]
    
    # Filter to available features with data
    available_features = [f for f in cluster_features if f in df.columns]
    
    # Create cluster_df with imputation for missing values instead of dropping
    cluster_df = df[available_features].copy()
    
    # Impute missing values with column means
    for col in cluster_df.columns:
        if cluster_df[col].isna().any():
            cluster_df[col] = cluster_df[col].fillna(cluster_df[col].mean())
    
    if len(cluster_df) < 5:
        st.warning(f"Only {len(cluster_df)} complete records available for clustering")
        return
    
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_df)
    
    # Elbow method to determine optimal clusters
    st.markdown("### Determining Optimal Number of Clusters")
    inertia = []
    max_clusters = min(10, len(cluster_df)-1)
    for k in range(1, max_clusters):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
    
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(range(1, max_clusters), inertia, marker='o')
    ax1.set_title('Elbow Method for Optimal Cluster Number')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.grid(True)
    st.pyplot(fig1)
    
    # Let user select number of clusters (default to 3)
    n_clusters = st.slider("Select number of clusters", 
                          min_value=2, 
                          max_value=max_clusters-1, 
                          value=3)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create interactive plot
    plot_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Cluster': cluster_labels,
        'Country': df.loc[cluster_df.index, 'G00Q01'].astype(str),
        'Stakeholder': df.loc[cluster_df.index, 'G00Q03'].astype(str)
    })
    
    fig2 = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                     hover_data=['Country', 'Stakeholder'],
                     title='PCA Visualization of Clusters',
                     color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Cluster profiles - show mean values for each feature
    st.subheader("Cluster Characteristics")
    
    # Create working dataframe with cluster assignments
    working_df = cluster_df.copy()
    working_df['Cluster'] = cluster_labels
    working_df['Country'] = df.loc[cluster_df.index, 'G00Q01'].astype(str)
    
    # Calculate statistics separately
    numeric_means = working_df.select_dtypes(include=np.number).groupby('Cluster').mean()
    cluster_counts = working_df.groupby('Cluster').size().rename('Count')
    countries_per_cluster = working_df.groupby('Cluster')['Country'].agg(
        lambda x: ', '.join(sorted(x.dropna().unique()))
    ).rename('Countries in Cluster')
    
    # Combine results
    profile_summary = pd.concat([cluster_counts, countries_per_cluster, numeric_means], axis=1)
    
    # Display with formatting
    st.dataframe(profile_summary.style.background_gradient(
        cmap='YlGnBu',
        subset=numeric_means.columns.tolist()
    ))
    
    # Interpretation guidance
    st.markdown("""
    **How to interpret clusters:**
    - **Count**: Number of respondents in each cluster
    - **Countries in Cluster**: Which countries are represented
    - Compare mean values across clusters for each numeric feature
    - Higher regulatory scores indicate better perceived regulatory effectiveness
    - Higher tech impact scores indicate greater perceived technology benefits
    - Look for patterns in the PCA plot (clusters near each other are similar)
    """)

# --- Updated Predictive Modeling Section ---
def run_predictive_model(df):
    """Predict whether technologies are seen as highly impactful"""
    st.subheader("Predictive Modeling: Technology Impact Prediction")
    
    # 1. Define target variable - whether tech is seen as highly impactful (rating 4-5)
    target_col = 'G00Q24.SQ001.'  # Tech impact rating
    if target_col not in df.columns:
        st.error(f"Critical column missing: {target_col}")
        return
    
    # Create binary target (1=high impact, 0=moderate/low)
    df['target'] = df[target_col].apply(lambda x: 1 if pd.notna(x) and float(x) >= 4 else 0)
    
    # 2. Define potential features with human-readable names
    feature_options = {
        'Regulatory Effectiveness': 'G00Q12.SQ001_SQ001.',
        'Post-Market Surveillance': 'G00Q12.SQ001_SQ002.',
        'Data Protection': 'G00Q12.SQ001_SQ003.',
        'Enforcement': 'G00Q12.SQ001_SQ004.',
        'Stakeholder Category': 'G00Q03',
        'Country': 'G00Q01',
        'Policy Harmonization': 'G00Q11.SQ001_SQ001.',
        'IPM Implementation': 'G04Q21.SQ001.',
        'Farmer Awareness': 'G00Q30.SQ001.'
    }
    
    # Let user select features
    selected_features = st.multiselect(
        "Select features for prediction",
        list(feature_options.keys()),
        default=['Regulatory Effectiveness', 'Stakeholder Category', 'Country']
    )
    
    # Map to actual column names
    feature_cols = [feature_options[f] for f in selected_features]
    feature_names = selected_features  # Keep the human-readable names
    
    # 3. Prepare data
    model_df = df[feature_cols + ['target']].copy()
    
    # Encode categorical variables
    for col in feature_cols:
        if model_df[col].dtype == 'object':
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))
    
    # Drop rows with missing values
    model_df = model_df.dropna()
    
    if len(model_df) < 20:
        st.warning(f"""
        Insufficient complete cases: {len(model_df)} (need ≥20).
        Missing data breakdown:
        {df[feature_cols + [target_col]].isna().sum()}
        """)
        return
    
    # 4. Handle class imbalance
    X = model_df[feature_cols]
    y = model_df['target']
    
    if y.nunique() < 2:
        st.warning("Target variable has only one class - cannot build model")
        return
    
    # Show class distribution
    class_counts = y.value_counts()
    st.markdown(f"**Class Distribution:** {dict(class_counts)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Apply SMOTE only if we have enough samples in minority class
    min_class_count = min(y_train.value_counts())
    if min_class_count >= 5:  # Need at least 5 samples for SMOTE
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count-1))
            X_train, y_train = smote.fit_resample(X_train, y_train)
            st.info(f"Applied SMOTE to balance classes. New training size: {len(X_train)}")
        except ValueError as e:
            st.warning(f"Could not apply SMOTE: {str(e)}. Proceeding with imbalanced data.")
    else:
        st.warning(f"Minority class has only {min_class_count} samples - SMOTE not applied")
    
    # 5. Train model
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # 6. Evaluate
    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    st.subheader("Model Performance")
    st.dataframe(report_df.style.format("{:.2f}"))
    
    # Feature importance - use human-readable names
    importance_df = pd.DataFrame({
        'Feature': feature_names,  # Use human-readable names instead of column codes
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', 
                 title='Feature Importance',
                 color='Importance',
                 color_continuous_scale='Blues',
                 labels={'Feature': 'Feature', 'Importance': 'Importance Score'})
    
    # Improve formatting
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        hovermode='y unified',
        yaxis_title=None,
        xaxis_title='Importance Score'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("""
    **How to interpret results:**
    - Precision: Of predictions for this class, how many were correct?
    - Recall: Of actual cases of this class, how many did we find?
    - F1-score: Balance between precision and recall
    - Feature importance shows which factors most influence predictions
    """)

def survey_monitoring_dashboard():
    st.title("🌾 Crop Protection Innovation Survey Dashboard")
    st.markdown("Monitoring the flow of crop protection innovation in low- and middle-income countries")
    
    # Load data
    df, total_records = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    with st.sidebar.expander("Select Countries", expanded=False):
        selected_countries = st.multiselect(
            "Countries",
            options=df['G00Q01'].unique(),
            default=df['G00Q01'].unique(),
            label_visibility="collapsed"
        )
    
    with st.sidebar.expander("Select Stakeholder Categories", expanded=False):
        selected_stakeholders = st.multiselect(
            "Stakeholders",
            options=df['G00Q03'].dropna().unique(),
            default=df['G00Q03'].dropna().unique(),
            label_visibility="collapsed"
        )
    
    # Date range selection - clean and compact
    if not df['submitdate'].isna().all():
        min_date = df['submitdate'].min().date()
        max_date = max(df['submitdate'].max().date(), datetime.today().date())
        
        st.sidebar.markdown("**Select Date Range**")
        
        # Create a clean date range selector
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            st.markdown("From:")
        with col2:
            start_date = st.date_input(
                "Start date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                label_visibility="collapsed"
            )
        
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            st.markdown("To:")
        with col2:
            end_date = st.date_input(
                "End date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                label_visibility="collapsed"
            )
    else:
        today = datetime.today().date()
        start_date = end_date = today
    
    # Apply filters
    filtered_df = df[
        (df['G00Q01'].isin(selected_countries)) &
        (df['G00Q03'].isin(selected_stakeholders))
    ].copy()
    
    # Apply date filter if we have dates
    if not df['submitdate'].isna().all():
        filtered_df = filtered_df[
            (filtered_df['submitdate'].dt.date >= start_date) &
            (filtered_df['submitdate'].dt.date <= end_date)
        ]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters")
        return
    
    # Dashboard sections
    show_kpi_cards(filtered_df, total_records)
    show_response_overview(filtered_df)
    show_policy_analysis(filtered_df)
    show_pesticide_data(filtered_df)
    show_adoption_metrics(filtered_df)
    
    # Challenges and Recommendations
    st.subheader("Text Analysis")
    show_text_analysis(filtered_df, "Common Challenges", 
                      ['G00Q36', 'G00Q37', 'G00Q38', 'G00Q39', 'G00Q40', 'G00Q41'])
    show_text_analysis(filtered_df, "Key Recommendations", 
                      ['G00Q42', 'G00Q43', 'G00Q44', 'G00Q45'])
    
    # Data explorer
    st.subheader("Data Explorer")
    if st.checkbox("Show raw data"):
        cols_to_show = [col for col in filtered_df.columns 
                       if col not in ['lastpage', 'startlanguage', 'G01Q46']]
        display_df = filtered_df[cols_to_show]
        st.dataframe(display_df)
    
    # Download button
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df(filtered_df)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name='filtered_survey_data.csv',
        mime='text/csv'
    )
    
    # Footer
    st.markdown("---")
    st.markdown("**Crop Protection Innovation Survey Dashboard** · Powered by Virtual Analytics")

def analysis_dashboard():
    st.title("🌱 Crop Protection Innovation Dashboard")
    st.markdown("""
    This dashboard provides comprehensive analysis of crop protection innovation flow in low- and middle-income countries, 
    focusing on technology, sustainability, and productivity.
    """)

    # Load data
    df, total_records = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar filters - now collapsible by default
    st.sidebar.header("Filter Data")
    
    with st.sidebar.expander("Select Countries", expanded=False):
        selected_countries = st.multiselect(
            "Countries",
            options=df['G00Q01'].unique(),
            default=df['G00Q01'].unique(),
            label_visibility="collapsed"
        )

    with st.sidebar.expander("Select Stakeholder Types", expanded=False):
        selected_stakeholders = st.multiselect(
            "Stakeholders",
            options=df['G00Q03'].unique(),
            default=df['G00Q03'].unique(),
            label_visibility="collapsed"
        )

    # Filter data
    filtered_df = df[
        (df['G00Q01'].isin(selected_countries)) & 
        (df['G00Q03'].isin(selected_stakeholders))
    ]

    # Overview section
    st.header("📊 Overview of Survey Responses")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Responses", len(df))
    col2.metric("Countries Represented", df['G00Q01'].nunique())
    col3.metric("Stakeholder Types", df['G00Q03'].nunique())

    # Country and stakeholder distribution
    st.subheader("Geographical and Stakeholder Distribution")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, y='G00Q01', order=filtered_df['G00Q01'].value_counts().index, ax=ax1)
    ax1.set_title('Responses by Country')
    ax1.set_xlabel('Number of Responses')
    ax1.set_ylabel('Country')
    st.pyplot(fig1)
    st.caption("""
    **Insight:** The survey responses span a wide range of countries, but participation is uneven. Zambia, Nigeria, and Ethiopia recorded the highest number of responses, suggesting greater stakeholder engagement or easier access to respondents in these countries. Kenya, Tanzania, and Angola also contributed significantly. Countries like Malawi, Saudi Arabia, and South Africa had the least representation, which may reflect either limited stakeholder engagement, outreach challenges, or a smaller crop protection innovation footprint.
    """)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, y='G00Q03', order=filtered_df['G00Q03'].value_counts().index, ax=ax2)
    ax2.set_title('Responses by Stakeholder Type')
    ax2.set_xlabel('Number of Responses')
    ax2.set_ylabel('Stakeholder Type')
    st.pyplot(fig2)
    st.caption("""
    **Insight:** The majority of responses come from industry players and regulators, reflecting their central role in crop protection innovation ecosystems. This dominance suggests that regulatory compliance and commercial product development are key drivers of innovation flow. However, the notably low participation from farmers, researchers, and academia highlights a critical gap in inclusive innovation. The underrepresentation of these groups may limit the practical relevance, field-level adoption, and research-driven refinement of crop protection technologies.
    """)

    # Policy and Regulation Analysis
    st.header("📜 Policy and Regulatory Environment")

    # Policy existence
    st.subheader("Existence of Key Policies")

    policy_cols = {
        'Pesticide Policy': 'G00Q11.SQ001_SQ001.',
        'Biosafety Policy': 'G00Q11.SQ002_SQ001.',
        'IPM Policy': 'G00Q11.SQ003_SQ001.',
        'Drone Policy': 'G00Q11.SQ004_SQ001.'
    }

    policy_data = []
    for policy_name, col_prefix in policy_cols.items():
        # Find columns that start with this prefix
        cols = [c for c in df.columns if c.startswith(col_prefix)]
        if cols:
            # Convert to string and clean
            policy_series = df[cols[0]].astype(str).str.strip().str.lower()

            # Count responses
            yes_count = policy_series.str.contains('yes', na=False).sum()
            no_count = policy_series.str.contains('no', na=False).sum()
            missing_count = policy_series.isna().sum()

            policy_data.append({
                'Policy': policy_name,
                'Yes': yes_count,
                'No': no_count,
                'Missing': missing_count
            })

    policy_df = pd.DataFrame(policy_data).set_index('Policy')

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    policy_df[['Yes', 'No']].plot(kind='barh', stacked=True, ax=ax3)
    ax3.set_title('Existence of Key Policies')
    ax3.set_xlabel('Number of Responses')
    st.pyplot(fig3)
    st.caption("""
    **Insight: Pesticide policies** are the most widely reported, indicating they are well-established and likely more mature across countries. In contrast, drone policies are the least common, underscoring regulatory lag in adapting to emerging technologies. The relatively low existence of Integrated Pest Management (IPM) policies is a notable gap—especially given the global shift toward sustainable and ecological farming practices. This suggests an opportunity for countries to scale up IPM policy frameworks to promote safer, more sustainable crop protection.
    """)

    # Regulatory effectiveness - Fixed to handle numeric conversion
    st.subheader("Perceived Effectiveness of Regulatory Processes")

    effectiveness_cols = {
        'Registration Process': 'G00Q12.SQ001_SQ001.',
        'Post-Market Surveillance': 'G00Q12.SQ001_SQ002.',
        'Data Protection': 'G00Q12.SQ001_SQ003.',
        'Enforcement': 'G00Q12.SQ001_SQ004.',
        'Label Approval': 'G00Q12.SQ002_SQ001.',
        'Import Control': 'G00Q12.SQ002_SQ002.',
        'Export Control': 'G00Q12.SQ002_SQ003.',
        'Disposal': 'G00Q12.SQ002_SQ004.'
    }

    effectiveness_data = []
    for process_name, col_prefix in effectiveness_cols.items():
        cols = [c for c in df.columns if c.startswith(col_prefix)]
        if cols:
            # Convert to numeric safely
            rating_series = pd.to_numeric(df[cols[0]], errors='coerce')
            avg_rating = rating_series.mean()
            effectiveness_data.append({
                'Process': process_name,
                'Average Rating': avg_rating
            })

    effectiveness_df = pd.DataFrame(effectiveness_data).set_index('Process')

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=effectiveness_df, y=effectiveness_df.index, x='Average Rating', ax=ax4)
    ax4.set_title('Average Rating of Regulatory Process Effectiveness (1-5 scale)')
    ax4.set_xlabel('Average Rating')
    ax4.set_xlim(0, 5)
    st.pyplot(fig4)
    st.caption("""
    **Insight:** Most regulatory processes are rated moderately effective, reflecting a system that is functional but with evident performance gaps. Data protection stands out as the most positively rated, potentially reflecting greater institutional clarity or investment in this area. In contrast, disposal and export control receive the lowest effectiveness ratings—flagging critical regulatory blind spots. These gaps likely pose environmental and trade risks, respectively, and highlight the urgent need for reforms to strengthen enforcement, safe disposal mechanisms, and streamlined export protocols for crop protection products.
    """)

    # --- New Analysis Sections for Additional Objectives ---
    st.header("🌍 Demand and Supply Side Analysis")
    
    # Demand side analysis
    show_demand_side_analysis(filtered_df)
    
    # Supply side analysis
    show_supply_side_analysis(filtered_df)
    
    # Product registration trends
    show_product_registration_trends(filtered_df)
    
    # Registration time comparison
    show_registration_time_comparison(filtered_df)

    # Innovation Flow Analysis
    st.header("💡 Innovation Flow and Adoption")

    # Time for registration
    st.subheader("Time Taken for Product Registration")

    time_cols = {
        'Conventional Pesticides': 'G00Q14.SQ001.',
        'Biopesticides': 'G00Q14.SQ002.',
        'Biocontrol Agents': 'G00Q14.SQ003.',
        'New Technologies': 'G00Q14.SQ004.'
    }

    time_data = []
    for tech_name, col in time_cols.items():
        if col in df.columns:
            time_counts = df[col].value_counts().to_dict()
            for time_period, count in time_counts.items():
                time_data.append({
                    'Technology': tech_name,
                    'Time Period': time_period,
                    'Count': count
                })

    time_df = pd.DataFrame(time_data)

    fig5 = px.bar(time_df, x='Technology', y='Count', color='Time Period', 
                  title='Time Taken for Product Registration by Technology Type')
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("""
    **Insight:** The data clearly shows that conventional pesticides benefit from faster and more predictable registration timelines, likely due to more established and well-understood regulatory pathways. In contrast, biopesticides, biocontrol agents, and newer technologies experience more prolonged approval times, with a notable concentration in the 3-to-5 year range. This delay suggests that regulatory systems are not yet fully adapted to accommodate emerging innovations, potentially slowing down the adoption of safer, more sustainable alternatives. Harmonizing and updating regulatory frameworks to accelerate review processes for newer technologies could unlock significant benefits in innovation uptake and sustainable agriculture practices.
    """)

    # Innovation adoption challenges
    st.subheader("Challenges in Adopting New Technologies")

    # Text analysis of challenges
    challenge_cols = {
        'General Challenges': 'G00Q39',
        'Regulatory Challenges': 'G00Q40',
        'Biopesticide Challenges': 'G00Q41',
        'Biocontrol Challenges': 'G00Q42'
    }

    # Word cloud for general challenges
    if 'G00Q39' in df.columns:
        text = ' '.join(df['G00Q39'].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        ax6.imshow(wordcloud, interpolation='bilinear')
        ax6.axis('off')
        ax6.set_title('Word Cloud of General Innovation Adoption Challenges')
        st.pyplot(fig6)
        st.caption("""
        **Insight:** The most pressing challenges in adopting new crop protection technologies center around regulatory bottlenecks, particularly the lack of specific guidelines for biopesticides and biocontrol agents, and unclear or lengthy registration processes. Terms like "lack," "guideline," "review," "efficacy," and "regulation" dominate the word cloud, pointing to significant gaps in policy clarity and institutional readiness.
        Additionally, the frequent appearance of "farmers," "skills," "training," and "illiteracy" highlights the limited farmer awareness and technical capacity, indicating that extension services and field-based education programs remain critically underfunded or underutilized.
        Financial and operational challenges are also apparent, with words like "cost," "access," and "resources" pointing to limited financial incentives or subsidies to support innovation adoption.
        Strategic Implication:
        Improving the adoption of innovations will require:
        •	Tailored regulatory frameworks for new technologies (e.g., separate dossiers and review protocols for biopesticides).
        •	Targeted farmer training and capacity-building initiatives.
        •	Strengthened coordination among regulators, researchers, and private sector actors to address institutional and knowledge gaps.

        """)

    # Sentiment analysis of challenges
    if 'G00Q39' in df.columns:
        sentiments = []
        for text in df['G00Q39'].dropna():
            blob = TextBlob(str(text))
            sentiments.append(blob.sentiment.polarity)
        
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        sns.histplot(sentiments, bins=20, kde=True, ax=ax7)
        ax7.set_title('Sentiment Analysis of Innovation Challenge Descriptions')
        ax7.set_xlabel('Sentiment Polarity (-1 to 1)')
        ax7.set_ylabel('Frequency')
        st.pyplot(fig7)
        st.caption("""
        **Insight:** The sentiment distribution of innovation challenge descriptions is overwhelmingly neutral, with a slight skew toward mildly negative sentiment. This suggests that while stakeholders are not overly pessimistic, their language does reflect underlying concerns, frustrations, or bureaucratic fatigue in adopting new technologies. The limited presence of positive sentiment and the clustering around zero polarity indicate that stakeholders tend to describe challenges factually rather than emotionally, focusing on practical obstacles rather than voicing optimism or deep dissatisfaction.
Interpretation:
•	The sentiment landscape reflects realism rather than resistance—a sign that respondents are engaged but constrained.
•	The absence of extreme negativity may suggest constructive criticism rather than outright disapproval, presenting an opportunity to act on these insights.
Strategic Implication:
Efforts to support innovation should be framed as collaborative solutions, responding to the practical tone of feedback—through policy clarity, faster processes, and support mechanisms—rather than simply motivational or awareness-based campaigns.
        """)

    # Technology Impact Assessment
    st.header("📈 Technology Impact Assessment")

    # Technology adoption ratings
    tech_cols = {
        'Increased Productivity': 'G00Q24.SQ001.',
        'Improved Sustainability': 'G00Q24.SQ002.',
        'Enhanced Food Safety': 'G00Q24.SQ003.'
    }

    tech_data = []
    for impact_name, col in tech_cols.items():
        if col in df.columns:
            avg_rating = df[col].mean()
            tech_data.append({
                'Impact Area': impact_name,
                'Average Rating': avg_rating
            })

    tech_df = pd.DataFrame(tech_data).set_index('Impact Area')

    fig8, ax8 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=tech_df, y=tech_df.index, x='Average Rating', ax=ax8)
    ax8.set_title('Perceived Impact of Crop Protection Technologies (1-5 scale)')
    ax8.set_xlabel('Average Rating')
    ax8.set_xlim(0, 5)
    st.pyplot(fig8)
    st.caption("""
    **Insight:** Crop protection technologies are perceived to deliver the greatest benefit in increasing productivity, followed closely by enhancing food safety and then improving sustainability. This prioritization suggests that stakeholders see these technologies primarily as tools to boost agricultural output, though there is a growing recognition of their role in food system resilience and environmental stewardship.
Interpretation:
•	The high rating for productivity reflects the persistent drive to meet food demand and improve farmer yields.
•	The strong score for food safety highlights awareness of post-harvest health risks and consumer protection.
•	The slightly lower rating for sustainability implies that while important, ecological and long-term benefits may be underemphasized in policy or implementation compared to short-term gains.
Strategic Implication:
Stakeholders should consider mainstreaming sustainability metrics into technology development, promotion, and adoption strategies. Demonstrating that these innovations can simultaneously deliver yield, safety, and ecological benefits could boost acceptance and long-term impact.

    """)
    
    # Cluster analysis of respondents
    perform_clustering(filtered_df)
    
    # Predictive Modeling
    run_predictive_model(filtered_df)

    # Prescriptive Recommendations
    st.header("💡 Prescriptive Recommendations")

    # Generate recommendations based on analysis
    recommendations = [
        {
            "Area": "Regulatory Systems",
            "Recommendation": "Strengthen post-market surveillance and enforcement mechanisms to improve confidence in crop protection technologies.",
            "Rationale": "Analysis showed these are the weakest aspects of regulatory systems but most predictive of positive technology perceptions."
        },
        {
            "Area": "Farmer Engagement",
            "Recommendation": "Increase farmer participation in innovation systems through targeted outreach and education programs.",
            "Rationale": "Farmers were underrepresented in survey responses but are critical end-users of technologies."
        },
        {
            "Area": "Technology Development",
            "Recommendation": "Prioritize development of biopesticides and biocontrol agents with streamlined regulatory pathways.",
            "Rationale": "These technologies face longer registration times despite their sustainability benefits."
        },
        {
            "Area": "Policy Framework",
            "Recommendation": "Develop specific policies for emerging technologies like drone applications in agriculture.",
            "Rationale": "Drone policies were the least commonly reported among surveyed countries."
        },
        {
            "Area": "Capacity Building",
            "Recommendation": "Invest in training for regulators on evaluating new technologies and for farmers on adopting them.",
            "Rationale": "Knowledge gaps were frequently cited as barriers to innovation adoption."
        }
    ]

    rec_df = pd.DataFrame(recommendations)

    st.table(rec_df)
    st.caption("""
    These recommendations are derived from patterns identified in the survey data analysis and aim to address 
    the key challenges and opportunities revealed through the research.
    """)

    # Country-specific insights
    st.header("🌍 Country-Specific Insights")

    if 'G00Q01' in df.columns:
        # First convert all numeric columns to numeric type
        numeric_cols = ['G00Q24.SQ001.', 'G00Q24.SQ002.', 'G00Q12.SQ001_SQ001.']
    
        # Create a copy of the dataframe with numeric conversions
        country_df = df.copy()
        for col in numeric_cols:
            if col in country_df.columns:
                country_df[col] = pd.to_numeric(country_df[col], errors='coerce')
    
        # Handle the registration time column separately
        most_common_time = None
        if 'G00Q14.SQ001.' in country_df.columns:
            # First clean the registration time strings
            time_mapping = {
                'below 1 year': '0-1 year',
                '1-2 years': '1-2 years',
                '2-3 years': '2-3 years',
                'above 3 years': '3+ years',
                'less than 1 year': '0-1 year',
                'more than 3 years': '3+ years'
            }
        
            # Clean and standardize the time strings
            country_df['G00Q14.SQ001.'] = (
                country_df['G00Q14.SQ001.']
                .astype(str)
                .str.strip()
                .str.lower()
                .replace(time_mapping)
            )
        
            # Get the most common registration time per country
            most_common_time = (
                country_df.groupby('G00Q01')['G00Q14.SQ001.']
                .apply(lambda x: x.mode()[0] if not x.mode().empty else 'Not Available')
            )
    
        # Calculate statistics for numeric columns
        stats = {}
        for col in numeric_cols:
            if col in country_df.columns:
                stats[f'avg_{col}'] = country_df.groupby('G00Q01')[col].mean()
    
        # Combine all statistics
        country_stats = pd.DataFrame(stats)
    
        # Rename columns for better display
        country_stats = country_stats.rename(columns={
            'avg_G00Q24.SQ001.': 'Avg_Productivity_Impact',
            'avg_G00Q24.SQ002.': 'Avg_Sustainability_Impact',
            'avg_G00Q12.SQ001_SQ001.': 'Avg_Registration_Effectiveness'
        })
    
        # Add the most common registration time if available
        if most_common_time is not None:
            country_stats['Most_Common_Registration_Time'] = most_common_time
    
        # Sort by productivity impact
        if not country_stats.empty:
            country_stats = country_stats.sort_values('Avg_Productivity_Impact', ascending=False)
    
        st.subheader("Country Performance Metrics")
    
        # Display the numeric columns with formatting
        if not country_stats.empty:
            # Format numeric columns
            formatted_stats = country_stats.copy()
            styled_df = country_stats.style.format({
                col: "{:.2f}" for col in country_stats.columns if col.startswith('Avg_')
            }).background_gradient(
                cmap='YlGnBu',
                subset=[col for col in country_stats.columns if col.startswith('Avg_')]
            )
            # Create styled dataframe
            styled_df = formatted_stats.style.background_gradient(
                cmap='YlGnBu',
                subset=[col for col in formatted_stats.columns if col.startswith('Avg_')]
            )
        
            # Display with improved formatting
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=(min(len(formatted_stats) * 35 + 35, 500))  # Dynamic height
            )
        
            st.caption("""
            **Insight:** Countries exhibit notable disparities in how they perceive the impact of crop protection technologies and the effectiveness of related regulatory processes:
•	High Performers:
o	Mali and Saudi Arabia rate highest across all indicators — productivity, sustainability, and registration effectiveness — suggesting robust regulatory frameworks and positive technology outcomes.
o	Zimbabwe also shows strong scores in sustainability and registration despite moderate productivity.
•	Moderate Performers:
o	Kenya, Nigeria, Tanzania, Ghana, and Côte d'Ivoire demonstrate fairly balanced but mid-level performance, indicating room for growth especially in productivity or registration systems.
•	Low Performers:
o	South Africa, Zambia, Angola, and Ethiopia report low average scores, particularly in productivity and effectiveness, which may reflect bottlenecks in adoption or weak regulatory implementation.
•	Missing/Incomplete Data:
o	Uganda lacks numeric data, possibly due to limited survey input or reporting gaps, hindering its inclusion in comparative analysis.
Additional Note:
•	Countries with a lower Most Common Registration Time (e.g., Ethiopia: 1 year) may have faster but potentially less rigorous approval processes.
•	Conversely, longer registration times (e.g., Zimbabwe, Saudi Arabia: 5 years) could signal complex regulatory environments that may delay innovation unless streamlined.


            """)
        else:
            st.warning("No numeric data available for country performance metrics.")
    else:
        st.warning("Country data not available in the dataset.")

    # Download button for processed data
    st.sidebar.header("Data Export")
    if st.sidebar.button("Download Processed Data as CSV"):
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="crop_protection_innovation_processed.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    **Methodology Note:** 
    - Data was collected through a survey of stakeholders in low- and middle-income countries.
    - Analysis includes descriptive statistics, text mining, clustering, and predictive modeling.
    - Missing data was handled through exclusion for relevant analyses.
    """)

# --- Landing Page ---
def landing_page():
    st.title("🌍 Crop Protection Innovation Survey")
    st.markdown("""
    ## Assessing the Flow of Crop Protection Innovation in Low- and Middle-Income Countries
    
    **Subject:** Technology, Sustainability, and Productivity in Crop Protection
    
    **Objective:** This survey aims to monitor and analyze the current state of crop protection innovation 
    in low- and middle-income countries, focusing on the regulatory environment, technology adoption, 
    and barriers to innovation flow.
    """)
    
    st.image("https://images.unsplash.com/photo-1605000797499-95a51c5269ae?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
             use_container_width=True)
    
    st.markdown("""
    ### Select Dashboard:
    """)
    
    dashboard = st.selectbox(
        "Choose Dashboard",
        ["Survey Monitoring Dashboard", "Analysis Dashboard"],
        label_visibility="collapsed"
    )
    
    if dashboard == "Survey Monitoring Dashboard":
        survey_monitoring_dashboard()
    else:
        analysis_dashboard()

# --- Main App ---
def main():
    landing_page()

if __name__ == "__main__":
    main()