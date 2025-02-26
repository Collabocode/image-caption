from .knowledge_bases import USD_TO_ANY_EXCHANGE_RATES_JSON, ANY_TO_USD_EXCHANGE_RATES_JSON

intro_Any_to_USD_currency = f'I have a JSON object: This is the JSON object that contains the exchange rate of any currencies to US Dollar: {ANY_TO_USD_EXCHANGE_RATES_JSON}. '


LLM_step_by_step_template = f'Here is the step-by-step procedure:\n\
    1. Get current currency to US Dollar exchange rates in float. The current currency can be obtained from the given statement. Please put in (exchange_rate)\n\
    2. Get the current value. The current value can be obtained from the given statement. Please put in (current_value)\n\
    3. Get US Dollar to target currency exchange rate in float. The target currency can be obtained from the JSON object that contains **USD to various currencies** from the given statement. Please put in (target_exchange_rate)'

LLM_instruct_currency_converter = 'This is not a python code. The desired output format should be in JSON format:\n\
        ''exchange_rate_to_usd'': exchange_rate_to_usd, ''current_value'': current_value, ''target_exchange_rate'': target_exchange_rate, ''current_currency'': current_currency, ''target_currency'': target_currency,.\
        Make sure to only return the JSON data. Ensure adherence to this format in your response; any other formats will no be accepted.'

Instruct_CoT_1_LLM = 'I need to give me the source currency, the value of the source currency, and the target currency. The respond format should be in JSON format:\n' +\
' ''source_currency'': source currency, ''value'': value, ''target_currency'': target currency. Make sure to only return the JSON data. Ensure adherence to this format in your response; any other formats will no be accepted. '



def generate_CoT_2_LLM(src_currency, tgt_currency):
    # I need to retrieve the exchange rate for {src_currency} in {ANY_TO_USD_EXCHANGE_RATES_JSON} and {tgt_currency}. 
    intro_CoT = f'Here i provide a set of currency name and its exchange rate in dictionary as follows: {ANY_TO_USD_EXCHANGE_RATES_JSON}'
    Instruct_CoT_2_LLM = intro_CoT + f'Now, i need to give me the source exchange rate for {src_currency} and then the target exchange rate for {tgt_currency} . The respond format should be in JSON format: ' +\
' ''source_exchange_rate'': exchange rate for source currency, ''target_exchange_rate'': exchange rate for target currency. Make sure to only return the JSON data. Ensure adherence to this format in your response; any other formats will no be accepted. '
    return Instruct_CoT_2_LLM
def generate_CoT_3_LLM(currency):
    Instruct_CoT_3_LLM = f'I need to identify the exchange rate for {currency} in {USD_TO_ANY_EXCHANGE_RATES_JSON}. This exchange rate can be obtained from the given JSON object. The respond format should be in JSON format:\n' +\
' ''target_exchange_rate'': exchange rate. Make sure to only return the JSON data. Ensure adherence to this format in your response; any other formats will no be accepted. '
    return Instruct_CoT_3_LLM

DS_money_changer_system_prompt = 'You are a Money Changer Bot, an automated service that converts any currency. \
    Your task is to identify the source currency, the source currency''s value, and the target currency from the user''s input. Afterwards, please get the exchange rate value from the given JSON object for both source currency and target currency. Always respond in JSON format.' + intro_Any_to_USD_currency

instruct_LLM_money_changer =  f'Now, i need to give me the source exchange rate and then the target exchange rate. The respond format should be in JSON format: ' +\
' ''source_exchange_rate'': exchange rate for source currency, ''target_exchange_rate'': exchange rate for target currency. Make sure to only return the JSON data. Ensure adherence to this format in your response; any other formats will no be accepted. '









# LLM_step_by_step_template = f'Here are the specific instructions:\n\
#     First, you need to identify the exchange rate for the current currency in relation to USD. This exchange rate can be obtained from the input statement. \
#     Convert this value into a floating-point number to make it usable for calculations.\
#     Store the result in a variable called exchange_rate.\n\
#     Get the Value in the Current Currency:\n\
#     From the same input statement, extract the value that represents the amount in the current currency (e.g., how much money is being converted).\n\
#     Again, store this value as a floating-point number in a variable called current_value.\n\
#     identify the value of target Currency Exchange Rate from USD to target currency from {USD_TO_ANY_EXCHANGE_RATES_JSON}\
#     Convert this value into a floating-point number to make it usable for calculations.\n\
#     Store this Target Currency Exchange Rate in a variable called target_currency.\n '
# LLM_instruct_currency_converter = 'Note: The desired output format should in JSON format: ''exchange_rate'': exchange_rate, ''current_value'': current_value, ''target_currency'': target_currency.\
#     Ensure adherence to this format in your response; any other formats will no be accepted'
# LLM_step_by_step_template = f'Here is the step-by-step procedure:\n\
#     1. Get current currency to US Dollar exchange rates in float. The current currency can be obtained from the given statement. Please put in (exchange_rate)\n\
#     2. Multiply the current value with the exchange_rate. The current value can be obtained from the given statement. Please put in (us_norm)\n\
#     3. Last, multiply us_norm with the value of target currency from this JSON {USD_TO_ANY_EXCHANGE_RATES_JSON}. Please put in (final_price). Note that this is the final result'

# LLM_instruct_currency_converter = 'Only return the JSON data. The desired output format should be in JSON format:\n\
#         ''exchange_rate'': exchange_rate, ''current_value'': current_value, ''target_currency'': target_currency.\
#         Make sure to only return the JSON data. Ensure adherence to this format in your response; any other formats will no be accepted'
# Currency: current_currency -> target currency\n\
#     price: final_price\n\
# 3. Last, multiply us_norm with 15135. Please put in (final_price). Note that this is the final result in indonesia currency'
# f'Step-by-Step Procedure:\n \
#     Extract the Current Currency Exchange Rate to USD:\n\
#     First, you need to identify the exchange rate for the current currency in relation to USD. This exchange rate can be obtained from the input statement you are working with. \
#     Convert this value into a floating-point number to make it usable for calculations.\
#     Store the result in a variable called exchange_rate.\n\
#     Get the Value in the Current Currency:\n\
#     From the same input statement, extract the value that represents the amount in the current currency (e.g., how much money is being converted).\n\
#     Again, store this value as a floating-point number in a variable called current_value.\n\
#     identify the exchange rate for target Currency Exchange Rate from USD to target currency from this JSON object: {USD_TO_ANY_EXCHANGE_RATES_JSON}:\n\
#     Convert this value into a floating-point number to make it usable for calculations.\n\
#     Store this Target Currency Exchange Rate in a variable called target_currency.\n\
#     Calculate the Final Converted Price:\
#     Using the values you have retrieved (exchange_rate, current_value, and target_currency), compute the final converted amount using the following formula: final_price = exchange_rate * current_value * target_currency\n\
#     The value stored in final_price will be the final converted amount from the current currency to the target currency.'