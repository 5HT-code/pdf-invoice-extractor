import streamlit as st
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from google.oauth2 import service_account
import json
import pandas as pd
import zipfile
import io
import os

service_account_info = st.secrets["gcp_service_account"]
service_account_json = json.dumps(service_account_info)

# Authenticate with service account
def authenticate_with_service_account(key_path):
    credentials = service_account.Credentials.from_service_account_file(key_path)
    vertexai.init(project="vision-419710", location="us-central1", credentials=credentials)

# Function to load and encode the PDF file
def load_pdf_file(file):
    pdf_data = file.read()
    encoded_pdf = base64.b64encode(pdf_data).decode('utf-8')
    return encoded_pdf

# Function to generate content using Vertex AI GenerativeModel
def generate_invoice_extraction(file):
    encoded_pdf = load_pdf_file(file)
    document1 = Part.from_data(mime_type="application/pdf", data=encoded_pdf)
    
    textsi_1 = """You are tasked with extracting and organizing data from invoice PDFs. Ensure you infer values based on context and provide all requested fields in a structured format, even if labeled differently."""
    
    text1 = """Please extract the following information from the attached invoice PDF and return the results in JSON format:
    Invoice Details:
    invoice_number: The invoice number (e.g., \"Invoice No.\", \"Inv#\", etc.).
    invoice_date: The date of the invoice (e.g., \"Date\", \"Invoice Date\").
    due_date: The payment due date (e.g., \"Due Date\").
    place_of_supply: The place of supply (e.g., \"recepient's state\").
    place_of_origin: The supplier's origin (e.g., supplier's state).
    receiver_name: The recipient\'s name (e.g., \"Billed To\").
    gstin_supplier: The supplier's GSTIN.
    gstin_recipient: The recipient\'s GSTIN, if available.
    taxable_value: Amount before tax and after discount.
    invoice_value: Amount after tax.
    tax_amount: total tax amount.
    Line Items:
    For each item, extract:
    item_name: The name or description of the item.
    rate_per_item_after_discount: The cost after discounts.
    quantity: The number of units/items.
    taxable_value: The taxable value of the item.
    sgst_amount, cgst_amount, igst_amount: Applicable tax amounts (if available).
    sgst_rate, cgst_rate, igst_rate: Applicable tax rates (if available).
    tax_amount: If CGST, SGST, and IGST amounts are not explicitly mentioned, extract the general tax amount.
    tax_rate: If CGST, SGST, and IGST rates are not explicitly mentioned, extract the general tax rate.
    final_amount: The final amount payable for the item.
    Total Summary:
    total_taxable_value: The sum of taxable values.
    total_cgst_amount, total_sgst_amount, total_igst_amount: Sum all CGST, SGST, and IGST amounts if multiple tax rates are mentioned. For example, if the invoice states \"CGST @ 6% = ₹X\" and \"CGST @ 9% = ₹Y\", sum the amounts (X + Y) for a consolidated total CGST amount. Apply the same rule for SGST and IGST if applicable. Follow this method strictly.
    total_tax_amount: If CGST, SGST, or IGST are not explicitly mentioned, extract and sum the general tax amount as total_tax_amount. Do not use this field if specific CGST, SGST, or IGST amounts are present.
    total_invoice_value: Total invoice value after taxes.
    rounding_adjustment: Any rounding adjustments.
    Return the extracted data in JSON format."""

    model = GenerativeModel("gemini-1.5-flash-002", system_instruction=textsi_1)

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0.2,
        "top_p": 0.95,
    }
    
    safety_settings = [
        SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
        SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
        SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
        SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    ]

    responses = model.generate_content(
        [text1, document1],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    return responses

def check_accuracy(invoices_df, items_df):
    invoices_df['invoice_value'] = pd.to_numeric(invoices_df['invoice_value'].astype(str).str.replace(',', ''), errors='coerce')
    invoices_df['taxable_value'] = pd.to_numeric(invoices_df['taxable_value'].astype(str).str.replace(',', ''), errors='coerce')
    
    items_df['final_amount'] = pd.to_numeric(items_df['final_amount'].astype(str).str.replace(',', ''), errors='coerce')
    items_df['taxable_value'] = pd.to_numeric(items_df['taxable_value'].astype(str).str.replace(',', ''), errors='coerce')

    invoices_invoice_value = invoices_df['invoice_value'].sum()
    invoices_taxable_value = invoices_df['taxable_value'].sum()

    items_invoice_value = items_df['final_amount'].sum()
    items_taxable_value = items_df['taxable_value'].sum()

    invoice_value_check = abs(invoices_invoice_value - items_invoice_value) < 1
    taxable_value_check = abs(invoices_taxable_value - items_taxable_value) < 1

    return invoice_value_check and taxable_value_check

def combine_invoices_and_items_df(invoices_df, items_df):
    new_df = items_df.copy()

    new_df['invoice_number'] = invoices_df['invoice_number'][0]
    new_df['invoice_date'] = invoices_df['invoice_date'][0]
    new_df['place_of_supply'] = invoices_df['place_of_supply'][0]
    new_df['place_of_origin'] = invoices_df['place_of_origin'][0]
    new_df['gstin_supplier'] = invoices_df['gstin_supplier'][0]
    new_df['gstin_recipient'] = invoices_df['gstin_recipient'][0]

    numeric_columns = ['taxable_value', 'sgst_amount', 'cgst_amount', 'igst_amount', 'sgst_rate', 'cgst_rate', 'igst_rate', 'tax_amount', 'tax_rate', 'final_amount']
    for col in numeric_columns:
        new_df[col] = pd.to_numeric(new_df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    new_df = new_df.drop(['item_name', 'rate_per_item_after_discount', 'quantity'], axis=1)

    return new_df

def process_file(file):
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}

    if file.name in st.session_state.processed_files:
        return st.session_state.processed_files[file.name]

    responses = generate_invoice_extraction(file)
    json_string = ''.join([response.text for response in responses])


    json_string = json_string.replace('```', '').replace('json\n', '')

    try:
        invoice_dict = json.loads(json_string)


        invoices_df = pd.DataFrame(invoice_dict['Invoice Details'], index=[0])
        items_df = pd.DataFrame(invoice_dict['Line Items'])



        accuracy_check = check_accuracy(invoices_df, items_df)

        if accuracy_check:
            new_df = combine_invoices_and_items_df(invoices_df, items_df)
            st.session_state.processed_files[file.name] = {'status': 'passed', 'data': new_df}
        else:
            st.session_state.processed_files[file.name] = {'status': 'failed', 'data': file}

    except json.JSONDecodeError:
        st.session_state.processed_files[file.name] = {'status': 'failed', 'data': file}

    return st.session_state.processed_files[file.name]

def clear_session():
    if 'uploaded_files' in st.session_state:
        for file in st.session_state.uploaded_files:
            file.close()
    st.session_state.clear()
    st.experimental_rerun()

def main():
    st.title("Invoice Data Extractor")

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files

    if st.session_state.uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_files = len(st.session_state.uploaded_files)
        processed_files = 0
        passed_files = 0
        failed_files = 0

        all_new_dfs = []
        failed_files_list = []

        for file in st.session_state.uploaded_files:
            processed_files += 1
            status_text.text(f"Processing file {processed_files} of {total_files}")

            result = process_file(file)

            if result['status'] == 'passed':
                passed_files += 1
                all_new_dfs.append(result['data'])
            else:
                failed_files += 1
                failed_files_list.append(result['data'])

            progress_bar.progress(processed_files / total_files)

        if all_new_dfs:
            combined_df = pd.concat(all_new_dfs, ignore_index=True)

            csv = combined_df.to_csv(index=False)
            st.download_button(
                label="Download CSV for passed files",
                data=csv,
                file_name="passed_invoices.csv",
                mime="text/csv",
            )

        if failed_files_list:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for file in failed_files_list:
                    zip_file.writestr(file.name, file.getvalue())
            
            st.download_button(
                label="Download ZIP of failed files",
                data=zip_buffer.getvalue(),
                file_name="failed_invoices.zip",
                mime="application/zip",
            )

        st.write(f"Total files: {total_files}")
        st.write(f"Files processed: {processed_files}")
        st.write(f"Files passed accuracy check: {passed_files}")
        st.write(f"Files failed accuracy check: {failed_files}")

        if st.button("Clear Session and Start Over"):
            clear_session()

if __name__ == "__main__":
    authenticate_with_service_account(service_account_json)
    main()
