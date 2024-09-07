import streamlit as st
import pdfplumber
import pandas as pd
import json
import os
import zipfile
from io import BytesIO

# Function to extract text from PDFs
def extract_from_pdf(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            all_text += page.extract_text() or ""
    return all_text

# Function to analyze extracted text with Groq (mock implementation)
def analyze_text_with_groq(text):
    # Replace with actual API call to Groq when integrating
    # Here, a mock JSON output is returned for demonstration
    prompt = combined_prompt_invoice.format(text=text)
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None

# Function to create DataFrames from JSON data
def create_dataframes(json_data):
    try:
        data = json.loads(json_data)
        df_items = pd.DataFrame(data['Items'])
        df_invoice = pd.DataFrame({
            'Invoice Number': [data.get('Invoice Number', '')],
            'Invoice Date': [data.get('Invoice date', '')],
            'Place Of Supply': [data.get('Place Of Supply', '')],
            'Place of Origin': [data.get('Place of Origin', '')],
            'Receiver Name': [data.get('Receiver Name', '')],
            'Taxable Value': [data.get('Taxable Value', 0)],
            'Cgst Amount': [data.get('Cgst Amount', 0)],
            'Sgst Amount': [data.get('Sgst Amount', 0)],
            'Invoice Value': [data.get('Invoice Value', 0)],
            'Rounding Adjustment': [data.get('Rounding Adjustment', 0)],
            'GSTIN/UIN of Supplier': [data.get('GSTIN/UIN of Supplier', '')],
            'GSTIN/UIN of Recipient': [data.get('GSTIN/UIN of Recipient', '')]
        })
        return df_items, df_invoice
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON: {e}")
        return None, None

# Function to check accuracy
def check_accuracy(df_invoice, df_items):
    sum_total_amount = df_items['Total Amount'].sum()
    sum_cgst_amount = df_items['Cgst amount'].sum()
    sum_sgst_amount = df_items['Sgst amount'].sum()
    invoice_total = df_invoice.at[0, 'Invoice Value']
    rounding_adjustment = df_invoice.at[0, 'Rounding Adjustment']
    sub_total = df_invoice.at[0, 'Taxable Value']
    cgst_total = df_invoice.at[0, 'Cgst Amount']
    sgst_total = df_invoice.at[0, 'Sgst Amount']
    invoice_total_check = abs((invoice_total - rounding_adjustment) - sum_total_amount) < 0.01
    sub_total_check = abs(sub_total - sum_total_amount) < 0.01
    cgst_total_check = abs(cgst_total - sum_cgst_amount) < 0.01
    sgst_total_check = abs(sgst_total - sum_sgst_amount) < 0.01
    return all([invoice_total_check, sub_total_check, cgst_total_check, sgst_total_check])

# Streamlit App Layout
st.title("Invoice Processing and Accuracy Check")
uploaded_files = st.file_uploader("Upload PDF invoices", type="pdf", accept_multiple_files=True)

if uploaded_files:
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    passed_files = []
    failed_files = []
    invoice_data = []
    item_data = []

    for i, file in enumerate(uploaded_files):
        extracted_text = extract_from_pdf(file)
        formatted_data = analyze_text_with_groq(extracted_text)

        if formatted_data:
            df_items, df_invoice = create_dataframes(formatted_data)
            if df_items is not None and df_invoice is not None:
                if check_accuracy(df_invoice, df_items):
                    passed_files.append(file.name)
                    invoice_data.append(df_invoice)
                    item_data.append(df_items)
                else:
                    failed_files.append(file.name)

        # Update Progress
        progress_bar.progress((i + 1) / total_files)
        st.write(f"Files processed: {i + 1}/{total_files}")

    # Zip passed and failed files
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for file_name in passed_files:
            zipf.writestr(f"passed/{file_name}", open(file_name, 'rb').read())
        for file_name in failed_files:
            zipf.writestr(f"failed/{file_name}", open(file_name, 'rb').read())
    
    st.download_button(
        label="Download Zipped Files",
        data=zip_buffer.getvalue(),
        file_name="processed_files.zip",
        mime="application/zip"
    )

    # Save and download CSV files
    if invoice_data and item_data:
        invoices_df = pd.concat(invoice_data, ignore_index=True)
        items_df = pd.concat(item_data, ignore_index=True)

        st.write("Invoices Data:")
        st.dataframe(invoices_df)

        st.write("Items Data:")
        st.dataframe(items_df)

        csv_invoice = invoices_df.to_csv(index=False).encode('utf-8')
        csv_items = items_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Invoices CSV",
            data=csv_invoice,
            file_name="invoices.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Download Items CSV",
            data=csv_items,
            file_name="items.csv",
            mime="text/csv"
        )

    # Summary of results
    st.write(f"Total files processed: {total_files}")
    st.write(f"Files passed accuracy check: {len(passed_files)}")
    st.write(f"Files failed accuracy check: {len(failed_files)}")
