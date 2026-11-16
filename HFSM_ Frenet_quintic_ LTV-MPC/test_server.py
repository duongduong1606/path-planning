import urllib.request

response = urllib.request.urlopen('http://127.0.0.1:8765/ui/app.js')
headers = dict(response.headers)

print("=== Cache Headers ===")
print(f"Cache-Control: {headers.get('Cache-Control', 'NOT SET')}")
print(f"ETag: {headers.get('ETag', 'NOT SET')}")

content = response.read().decode('utf-8')
print(f"\n=== Content Check ===")
print(f"Has rotate(yaw): {content.count('rotate(yaw)')} occurrences")
print(f"Has rotate(-yaw): {content.count('rotate(-yaw)')} occurrences")
print(f"Has 'h / 2 + (y - view.cy)': {('h / 2 + (y - view.cy)' in content)}")
print(f"Has 'h / 2 - (y - view.cy)': {('h / 2 - (y - view.cy)' in content)}")
