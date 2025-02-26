def rewrite_final_currency_output(meta_data, final_price):
    curr_currency = meta_data['source_currency']
    target_currency = meta_data['target_currency']
    source_exchange_rate = meta_data['source_exchange_rate']
    try:
        curr_value = meta_data['source_value']
    except:
        curr_value = meta_data['value']
    target_exchange_rate = float(1 /meta_data['target_exchange_rate'])
    final_price = "{:0,.2f}".format(float(final_price))
    if source_exchange_rate >0.1:
        kurs =  "{:0,.2f}".format(float(source_exchange_rate * target_exchange_rate)) 
    else:
        kurs =  "{:0,.5f}".format(float(source_exchange_rate * target_exchange_rate))
    respond = f'Current kurs from {curr_currency} to {target_currency}: {kurs} \n ' + \
    f'{curr_value} {curr_currency} to {target_currency}: {final_price} in {target_currency}'
    return respond