import sounddevice as sd

print("=" * 80)
print("DEFAULT DEVICES")
print("=" * 80)
print(sd.default.device)

print()

print("=" * 80)
print("AVAILABLE DEVICES")
print("=" * 80)

for i, device in enumerate(sd.query_devices()):
    print(f"{i}: {device['name']}")