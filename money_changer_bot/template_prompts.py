from deep_translator import GoogleTranslator
# Use any translator you like, in this example GoogleTranslator
translated = GoogleTranslator(source='en', target='id').translate("i'll explode your head")
print(translated)
# system_prompt = 'You are a helpful assistant'
