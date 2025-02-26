from loguru import logger
from .knowledge_bases import ANY_TO_USD_EXCHANGE_RATES_JSON #, USD_TO_ANY_EXCHANGE_RATES_JSON

# from .knowledge_bases import USD_TO_ANY_EXCHANGE_RATES_JSON

GPT_intro_currency = f'I have a JSON object: This is the JSON object that contains the exchange rate of any currencies to US Dollar: {ANY_TO_USD_EXCHANGE_RATES_JSON} (named as ANY_TO_USD).'
    # f'This is the JSON object that contains the exchange rate of US Dollar to any currencies: {USD_TO_ANY_EXCHANGE_RATES_JSON} (named as USD_TO_ANY).'

    # and the other contains exchange rates from USD to other currencies. Given a statement that specifies a source currency and its value, \
    # i want to get source currency and its value, target currency, exchange rates to USD, and exchange rate from USD to other currencies. '
GPT_instruction_currency = 'Now i need to first give me the name of the source currency, its value, and the target currency from the content, and then find the source' + \
    f'currency''s exchange rate in ANY_TO_USD (exchange_rate_to_usd), and then find the target currency''s exchange rate in ANY_TO_USD (target_exchange_rate). ' 



# GPT_instruction_currency = 'Step-by-Step Procedure:\n' + 'First, Extract the Source Currency''s Exchange Rate to USD:\n' + \
#     f'From the {ANY_TO_USD_EXCHANGE_RATES_JSON} JSON object, retrieve the exchange rate for the source_currency in relation to USD\n' + \
#     'Store this value as a floating-point number in the variable exchange_rate_to_usd.\n' + \
#     'From the given statement, retrieve the value associated with the source_currency.' + \
#     'Store this value as a floating-point number in the variable current_value.\n' + \
#     f'Last, From the {USD_TO_ANY_EXCHANGE_RATES_JSON} JSON object, retrieve the target exchange rate and ' +\
#     'Store this value in the variable target_exchange_rate.\n'
logger.info(f'GPT INFO: {GPT_instruction_currency}')
GPT_examples = 'For example: \n' + \
                'Input: Source Currency: British Pound, Value: 200, Target Currency: Euro\n' + \
                'any_to_usd_exchange_rates = {"British Pound": 1.339657}' + 'usd_to_any_exchange_rates = {"Euro": 0.894436}\n'+ \
                'Output: {"exchange_rate": 1.339657, "current_value": 200, "target_exchange_rate": 0.894436, "current_currency": "British Pound", "target_currency": "Euro"}'

# GPT_instruction_currency = f'Step-by-Step Procedure:\n\
#     First, Extract the Source Currency''s Exchange Rate to USD:\n\
#     From the {ANY_TO_USD_EXCHANGE_RATES_JSON} JSON object, retrieve the exchange rate for the source_currency in relation to USD \n\
#     Store this value as a floating-point number in the variable exchange_rate.\n\
#     Second, Extract the Value in the Source Currency:\n\
#     From the given statement, retrieve the value associated with the source_currency. \
#     Store this value as a floating-point number in the variable current_value.\n\
#     Third, Retrieve the Target Currency''s Exchange Rate from USD:\n\
#     From the {USD_TO_ANY_EXCHANGE_RATES_JSON} JSON object, find the exchange rate for converting USD to the target_currency. \
#     Store this value in the variable target_exchange_rate.\n'