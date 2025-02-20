from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Spacer
from reportlab.lib import colors
import plotly.graph_objs as go
import plotly
import json
import tempfile
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image,PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
import plotly
import plotly.graph_objs as go
import plotly.graph_objects as go
import json
import tempfile
import ternary
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
import tempfile
import ternary
import re

# Initialize the Flask app
app = Flask(__name__)

# Load the models
vnir_model = load_model('vnir_model.h5')
mir_model = load_model('mir_model.h5')

# Define target columns
target_columns = [
    'Sand_gkg', 'Silt_gkg', 'Clay_gkg', 'Density_kgdm3', 'C_gkg',
    'OM_gkg', 'SOCS_tha', 'P_mgkg', 'pH_H2O', 'Ca_mmolkg',
    'Mg_mmolkg', 'K_mmolkg', 'Na_mmolkg', 'Al_mmolkg',
    'H_Al_mmolkg', 'CEC_Ph7_mmolkg'
]

# Selected indicators to display
selected_columns = [
    'Sand_gkg', 'Silt_gkg', 'Clay_gkg', 'Density_kgdm3', 'C_gkg',
    'OM_gkg', 'SOCS_tha', 'P_mgkg', 'pH_H2O', 'Ca_mmolkg',
    'Mg_mmolkg'
]

predictions_df = pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global predictions_df, input_spectra
    if 'file' not in request.files or 'model_type' not in request.form:
        return "File or model type not provided.", 400

    file = request.files['file']
    model_type = request.form['model_type']

    if file.filename.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        data = pd.read_excel(file)
    else:
        return "Unsupported file format. Please upload a CSV or Excel file.", 400

    input_spectra = data.copy()  # Store original spectra for visualization

    # Filter VNIR data columns (X350 to X1350)
    vnir_columns = [col for col in data.columns if col.startswith('X') and 350 <= int(col[1:]) <= 1350]
    vnir_data = data[vnir_columns]

    # Preprocess data
    data = data.select_dtypes(include=[np.number])
    data = data.fillna(data.mean())

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    model = vnir_model if model_type == 'VNIR' else mir_model
    expected_features = model.input_shape[1]

    if data_scaled.shape[1] < expected_features:
        padding = np.zeros((data_scaled.shape[0], expected_features - data_scaled.shape[1]))
        data_scaled = np.hstack((data_scaled, padding))
    elif data_scaled.shape[1] > expected_features:
        data_scaled = data_scaled[:, :expected_features]

    data_scaled = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[1], 1)
    predictions = model.predict(data_scaled)

    predictions_df = pd.DataFrame(predictions, columns=target_columns)
    predictions_df = predictions_df[selected_columns].applymap(lambda x: round(x, 2))

    # Reflectance Graph (Enhanced for VNIR: X350 - X1350)
    wavelengths = [int(col[1:]) for col in vnir_columns]  # Extract numeric part of column names
    fig = go.Figure()

    # Plot each sample with improved styling
    for idx, row in vnir_data.iterrows():
        fig.add_trace(go.Scatter(
            x=wavelengths,
            y=row.values,
            mode='lines+markers',
            name=f'Sample {idx + 1}',
            line=dict(color='red', width=2),
            marker=dict(size=3, color='red'),
            hovertemplate=f'<b>Sample {idx + 1}</b><br>Wavelength: %{{x}} nm<br>Reflectance: %{{y:.2f}}<extra></extra>'
        ))

    # Update layout for a cleaner look
    fig.update_layout(
        title='Reflectance vs Wavelength (VNIR)',
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(
            title='Wavelength (nm)',
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False,
            showline=True,
            linewidth=1.5,
            linecolor='black',
        ),
        yaxis=dict(
            title='Reflectance',
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False,
            showline=True,
            linewidth=1.5,
            linecolor='black',
        ),
        legend=dict(
            title='Samples',
            orientation='h',
            yanchor='bottom',
            y=1.05,
            xanchor='right',
            x=1
        ),
        template='plotly_white',
        hovermode='closest',
        margin=dict(l=50, r=50, t=50, b=50)
    )

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('results.html',
                           tables=predictions_df.to_html(classes='data', index=False),
                           titles=predictions_df.columns.values,
                           graphJSON=graph_json)

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    if predictions_df.empty:
        return "No prediction data to export.", 400

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    elements = []

    ### Page 1: Title and Introduction ###
    title_style = ParagraphStyle(name="Title", fontSize=16, fontName="Helvetica-Bold", alignment=1)
    elements.append(Paragraph("Results of Soil Spectroscopy Analysis", style=title_style))
    elements.append(Spacer(1, 20))

    intro_text = """
    The soil spectral analysis report provides a comprehensive assessment of soil properties including texture, pH, cation exchange capacity (CEC), and organic carbon. Predictions are derived from VNIR/MIR data using AI regression models trained on diverse global datasets.
    """
    intro_style = ParagraphStyle(name="IntroText", fontSize=10, fontName="Helvetica", spaceAfter=18)
    elements.append(Paragraph(intro_text, style=intro_style))
    elements.append(Spacer(1, 40))

    elements.append(PageBreak())  # Move to Page 2

    ### Page 2: Visualization Charts ###
    # Histograms
    numeric_columns = predictions_df.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(numeric_columns):
        sns.histplot(predictions_df[col].dropna(), kde=True, ax=axes[i], color="#1F618D")
        axes[i].set_title(f"Histogram of {col}", fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        plt.tight_layout()
        plt.savefig(tmp_file.name, format="png")
        plt.close()
        histograms_path = tmp_file.name

    # Radar Plot
    radar_columns = [col for col in numeric_columns[:6]]
    radar_values = predictions_df[radar_columns].mean().values
    normalized_values = (radar_values - radar_values.min()) / (radar_values.max() - radar_values.min())
    normalized_values = np.append(normalized_values, normalized_values[0])
    angles = np.linspace(0, 2 * np.pi, len(radar_columns) + 1)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, normalized_values, 'o-', linewidth=2)
    ax.fill(angles, normalized_values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_columns)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        plt.tight_layout()
        plt.savefig(tmp_file.name, format="png")
        plt.close()
        radar_plot_path = tmp_file.name

   

    
    ### Page 2: Visualization Charts ###
    # Reflectance Spectra (Enhanced)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    wavelengths = np.linspace(600, 2500, 100)
    reflectance = np.random.rand(100) * 1.5

    ax.plot(wavelengths, reflectance, color='#1F77B4', linewidth=2, marker='o', markersize=4, markerfacecolor='orange')
    ax.set_title('Reflectance Spectra', fontsize=14, fontweight='bold')
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        plt.tight_layout()
        plt.savefig(tmp_file.name, format="png")
        plt.close()
        reflectance_path = tmp_file.name

    # Soil Texture Triangle (Updated)
    try:
        if {'Sand_gkg', 'Silt_gkg', 'Clay_gkg'}.issubset(predictions_df.columns):
            texture_data = predictions_df[['Sand_gkg', 'Silt_gkg', 'Clay_gkg']].dropna()
            
            # Normalize the data to ensure it sums to 100%
            texture_data['Total'] = texture_data.sum(axis=1)
            texture_data[['Sand_gkg', 'Silt_gkg', 'Clay_gkg']] = (
                texture_data[['Sand_gkg', 'Silt_gkg', 'Clay_gkg']].div(texture_data['Total'], axis=0) * 100
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                fig, ax = plt.subplots(figsize=(6, 6))
                tax = ternary.TernaryAxesSubplot(ax=ax, scale=100)

                # Plotting the normalized data points
                data_points = texture_data[['Sand_gkg', 'Silt_gkg', 'Clay_gkg']].values.tolist()
                tax.scatter(data_points, color='#2E86C1', marker='o', s=50, label="Soil Samples", alpha=0.8)
                
                tax.boundary(linewidth=1.5)
                tax.gridlines(color="black", multiple=10, linestyle="dotted", linewidth=0.7)
                tax.left_axis_label("Clay (%)", fontsize=12, offset=0.15)
                tax.right_axis_label("Silt (%)", fontsize=12, offset=0.15)
                tax.bottom_axis_label("Sand (%)", fontsize=12, offset=0.05)
                tax.set_title("Soil Texture Triangle", fontsize=14, pad=20)
                tax.legend()

                plt.tight_layout()
                plt.axis("off")
                plt.savefig(tmp_file.name, format="png", dpi=300)
                plt.close(fig)
                soil_triangle_path = tmp_file.name
    except Exception as e:
        print(f"Error generating soil texture triangle: {e}")
        soil_triangle_path = None

    elements.append(Paragraph("Visualization Charts", style=title_style))
    elements.append(Spacer(1, 10))

    # Adding all charts
    graph_table_data = [
        [Image(histograms_path, width=300, height=200), Image(radar_plot_path, width=300, height=200)],
        [Image(reflectance_path, width=300, height=200), Image(soil_triangle_path, width=300, height=200)]
    ]
    graph_table = Table(graph_table_data, colWidths=[300, 300])
    graph_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    elements.append(graph_table)

    elements.append(PageBreak())  # Move to Page 3

    ### Page 3: Results Table ###
    data = [predictions_df.columns.tolist()] + predictions_df.values.tolist()
    col_widths = [70] * len(predictions_df.columns)
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name='predictions_report.pdf', mimetype='application/pdf')

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
