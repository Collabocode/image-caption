GPT_output_template = {
    "status": "payment_status",
    "bank_name": "bank_name",
    "transaction_id": "transaction_id_value",
    "amount": "payment_amount",
    "currency": "payment_currency",
    "payment_date": "payment_date",
    "payment_time": "payment_time,",
    'admin_fee': 'admin_fee',
    "recipient": "recipient_details",
    "reference_number": "reference_number"

    # "message": "confirmation_message"
}
examples = {
    # "bank_name": "BCA",
    "transaction_id": "150741147",
    "status": "BERHASIL",
    "amount": "95000",
    "currency": "Indonesia IDR",
    "payment_date": "27/09/2024",
    "recipient": "MICHAEL BENEDICT",
    # "message": ""
}
VLLM_instruct_system ='You are a helpful assistant. '
VLLM_instruct_system_payment_verification = 'You are a Payment Verification Bot. Your task is to extrat payments information based on \
    the provided details such as status, bank name, transaction ID, payment amount, currency, payment date, payment_time, admin_fee, recipient details, and reference number. The recipient details can be platform name or a person name. Respond with an appropriate message.' +\
    'Always respond in a structured JSON format as follows:' + \
    f'{GPT_output_template}. '

VLLM_output_template = 'Always respond in a structured JSON format as follows:' + f'{GPT_output_template}. '

GPT_VLLM_System_instruction = 'Task: Understand and analyze the text and context of the provided image.' + \
    'Step 1: Carefully extract all relevant information from the image.'+ \
    'Step 2: Based on the extracted information, expand on the content where necessary to provide a complete and coherent response.' + \
    'Step 3: Ensure your response is structured in JSON format, with fields such as "transaction_id", "status", "amount", "currency", "payment_date", and others based on the content of the image.' + 'Reminder: Provide the response only in JSON format.'


GPT_JSON_FROMAT = {
  "transaction_id": "string",            #// Transaction reference number
  "status": "string",                    #// Status of the transaction, e.g., 'BERHASIL', 'GAGAL'
  "bank_name": "string",                 #// Name of the bank, usually indicated by a watermark or text in the image
  "amount": {
    "value": "number",                   #// Transaction amount in numeric form
    "currency": "string"                 #// Currency code, e.g., "IDR" for Indonesian Rupiah
  },
  "payment_date": "string",              #// Date and time of the transaction in ISO 8601 format or any other readable format
  "recipient": {
    "name": "string",                    #// Name of the recipient or related information, if available
    "account": "string"                  #// Recipient account information (partially hidden in many cases)
  },
  "admin_fee": "number",                 #// Admin fee, if present, otherwise 0
  "additional_details": "string",        #// Any additional information like NPWP, or other metadata included in the transaction
  "reference_code": "string"             #// Unique reference code of the transaction
}


SYSTEM_PROMPT_FROM_CC_TOOLS = 'You are a helpful assistant\n\nTemplate: {\n  "transaction_id": "string",            // Transaction reference number or start with \'Ke\'\n  "status": "string",                    // Status of the transaction, e.g., \'BERHASIL\', \'GAGAL\'.\n  "bank_name": "string",                 // Name of the bank, usually indicated by a watermark or text in the image\n  "amount": {\n    "value": "number",                   // Transaction amount in numeric form\n    "currency": "string"                 // Currency code, e.g., "IDR" for Indonesian Rupiah\n  },\n  "payment_date": "string",              // Date and time of the transaction in ISO 8601 format or any other readable format\n  "recipient": {\n    "name": "string",                    // Name of the recipient or related information, if available\n    "account": "string"                  // Recipient account information (partially hidden in many cases)\n  },\n  "admin_fee": "number",                 // Admin fee, if present, otherwise 0\n  "additional_details": "string",        // Any additional information like NPWP, or other metadata included in the transaction\n  "reference_code": "string"             // Unique reference code of the transaction\n}'
