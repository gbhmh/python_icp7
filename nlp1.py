import requests
from bs4 import BeautifulSoup

# Set the URL you want to webscrape from
url = 'https://en.wikipedia.org/wiki/Google'

# Connect to the URL
response = requests.get(url)

# Parse HTML and save to BeautifulSoup object¶
soup = BeautifulSoup(response.text, "html.parser")



# finding text content
text = soup.find_all(text=True)

output = ''
blacklist = [
	'[document]',
	'noscript',
	'header',
	'html',
	'meta',
	'head',
	'input',
	'script',
  'style',
]


for t in text:
	if t.parent.name not in blacklist:
		output += '{} '.format(t)
		a = t.encode("utf-8").rstrip()
		print("{}".format(a.decode("utf-8")), file=open("input.txt","a"))

